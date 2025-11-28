"""
Text embedding and similarity computation module
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from pathlib import Path
import pickle


class EmbeddingManager:
    """텍스트 임베딩 및 유사도 계산 관리 클래스"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Path = None):
        """
        Args:
            model_name: Sentence Transformer 모델 이름
            cache_dir: 임베딩 캐시 저장 디렉토리
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)
        
        # 캐시 설정
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".dataset_cache"
        cache_dir.mkdir(exist_ok=True)
        
        self.label_cache_file = cache_dir / "label_embeddings.pkl"
        self.folder_cache_file = cache_dir / "folder_embeddings.pkl"
        
        # 메모리 캐시
        self.label_cache = self._load_label_cache()
    
    def _load_label_cache(self) -> Dict[str, torch.Tensor]:
        """라벨 임베딩 캐시 로드"""
        if self.label_cache_file.exists():
            try:
                with open(self.label_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                print(f"[Embedding] Loaded {len(cache)} label embeddings from cache")
                return cache
            except:
                return {}
        return {}
    
    def save_label_cache(self):
        """라벨 임베딩 캐시 저장"""
        try:
            with open(self.label_cache_file, 'wb') as f:
                pickle.dump(self.label_cache, f)
        except Exception as e:
            print(f"Warning: Failed to save label cache: {e}")
    
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """
        텍스트 리스트를 임베딩
        
        Args:
            texts: 임베딩할 텍스트 리스트
        
        Returns:
            정규화된 임베딩 텐서
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
        # 정규화
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings
    
    def embed_texts_with_cache(self, texts: List[str]) -> torch.Tensor:
        """
        텍스트 리스트를 임베딩 (캐싱 사용)
        
        Args:
            texts: 임베딩할 텍스트 리스트
        
        Returns:
            정규화된 임베딩 텐서
        """
        # 캐시에서 찾을 수 없는 텍스트만 추출
        uncached_texts = []
        
        for text in texts:
            if text not in self.label_cache:
                uncached_texts.append(text)
        
        # 캐시되지 않은 텍스트만 임베딩
        if uncached_texts:
            print(f"  [Embedding] Computing embeddings for {len(uncached_texts)} new labels")
            new_embeddings = self.embed_texts(uncached_texts)
            
            # 캐시에 저장
            for text, embedding in zip(uncached_texts, new_embeddings):
                self.label_cache[text] = embedding.cpu()  # CPU로 이동해서 메모리 절약
        
        # 모든 텍스트의 임베딩을 캐시에서 가져오기
        result_embeddings = []
        for text in texts:
            result_embeddings.append(self.label_cache[text].to(self.device))
        
        return torch.stack(result_embeddings)
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        코사인 유사도 계산
        
        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩
        
        Returns:
            코사인 유사도 값 (0-1)
        """
        similarity = torch.cosine_similarity(embedding1, embedding2, dim=-1)
        return similarity.item()
    
    def get_folder_embeddings(self, dataset_dir: Path, use_cache: bool = True) -> Dict[Path, torch.Tensor]:
        """
        dataset 내 모든 폴더명의 임베딩 계산 (캐싱)
        
        Args:
            dataset_dir: 데이터셋 디렉토리
            use_cache: 캐시 사용 여부
        
        Returns:
            {폴더_경로: 임베딩} 딕셔너리
        """
        from utils import compute_dataset_hash
        
        # 캐시 확인
        if use_cache and self.folder_cache_file.exists():
            try:
                # dataset 해시 확인
                current_hash = compute_dataset_hash(dataset_dir)
                
                with open(self.folder_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # 해시가 같으면 캐시 사용
                if cached_data.get('dataset_hash') == current_hash:
                    print(f"[Embedding] Loaded folder embeddings from cache")
                    return cached_data['folder_embeddings']
            except:
                pass  # 캐시 로드 실패 시 재계산
        
        print(f"[Embedding] Computing folder embeddings...")
        
        # 모든 폴더 찾기 (dataset_dir의 직접 하위 폴더들)
        folders = []
        for item in dataset_dir.iterdir():
            if item.is_dir():
                folders.append(item)
        
        # 폴더명 임베딩 계산
        folder_embeddings = {}
        if folders:
            folder_names = [folder.name for folder in folders]
            embeddings = self.embed_texts(folder_names)
            
            for folder, embedding in zip(folders, embeddings):
                folder_embeddings[folder] = embedding
        
        # 캐시 저장
        if use_cache:
            try:
                current_hash = compute_dataset_hash(dataset_dir)
                with open(self.folder_cache_file, 'wb') as f:
                    pickle.dump({
                        'dataset_hash': current_hash,
                        'folder_embeddings': folder_embeddings
                    }, f)
                print(f"[Embedding] Saved folder embeddings to cache ({len(folder_embeddings)} folders)")
            except:
                pass
        
        return folder_embeddings
    
    def rank_folders_by_similarity(
        self,
        broad_categories: List[str],
        folder_embeddings: Dict[Path, torch.Tensor]
    ) -> List[tuple]:
        """
        대분류 카테고리와 폴더명의 유사도를 계산하여 정렬
        
        Args:
            broad_categories: 대분류 카테고리 리스트 (OR 관계)
            folder_embeddings: {폴더_경로: 임베딩} 딕셔너리
        
        Returns:
            [(폴더_경로, 최대_유사도)] 리스트 (유사도 높은 순으로 정렬)
        """
        if not folder_embeddings:
            return []
        
        # 대분류 임베딩
        category_embeddings = self.embed_texts(broad_categories)
        
        # 각 폴더의 최대 유사도 계산
        folder_scores = []
        for folder_path, folder_emb in folder_embeddings.items():
            max_similarity = 0.0
            
            # 모든 대분류와 비교해서 최대값 사용 (OR 관계)
            for cat_emb in category_embeddings:
                similarity = self.compute_similarity(folder_emb.unsqueeze(0), cat_emb.unsqueeze(0))
                max_similarity = max(max_similarity, similarity)
            
            folder_scores.append((folder_path, max_similarity))
        
        # 유사도 높은 순으로 정렬
        folder_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n[Folder Ranking] Top folders by similarity:")
        for folder, score in folder_scores[:10]:  # 상위 10개만 출력
            print(f"  {folder.name}: {score:.3f}")
        
        return folder_scores
