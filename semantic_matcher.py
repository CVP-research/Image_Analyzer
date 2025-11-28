"""
Semantic background matching module
의미론적으로 적합한 배경 위치를 찾는 모듈
"""

from pathlib import Path
from PIL import Image
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from segment import run_segmentation
from embedding import EmbeddingManager
from utils import get_image_cache_key
import pickle


class SemanticMatcher:
    """의미론적 배경 매칭 클래스"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        cache_dir: Path = None
    ):
        """
        Args:
            similarity_threshold: 유사도 임계값
            cache_dir: 캐시 디렉토리
        """
        self.similarity_threshold = similarity_threshold
        
        # 캐시 설정
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".dataset_cache"
        cache_dir.mkdir(exist_ok=True)
        
        self.cache_dir = cache_dir
        self.segment_labels_cache_dir = cache_dir / "segment_labels"
        self.segment_labels_cache_dir.mkdir(exist_ok=True)
        
        # 임베딩 매니저 초기화
        self.embedding_manager = EmbeddingManager(cache_dir=cache_dir)
    
    def get_segment_labels(self, image_path: Path, use_cache: bool = True) -> List[str]:
        """
        이미지의 segment 라벨들만 반환 (캐싱)
        
        Args:
            image_path: 이미지 파일 경로
            use_cache: 캐시 사용 여부
        
        Returns:
            segment 라벨 리스트 (예: ["chair", "table", "person"])
        """
        # 캐시 확인
        if use_cache and image_path is not None:
            cache_key = get_image_cache_key(image_path)
            cache_file = self.segment_labels_cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        labels = pickle.load(f)
                    print(f"  [Cache] Loaded segment labels from cache")
                    return labels
                except:
                    pass  # 캐시 로드 실패 시 재계산
        
        # Segmentation 수행하여 라벨만 추출
        image_pil = Image.open(image_path).convert("RGB")
        annotations, labeled_results = run_segmentation(image_pil)
        
        labels = [r["label"] for r in labeled_results]
        
        # 캐시 저장 (라벨만)
        if use_cache and image_path is not None:
            cache_key = get_image_cache_key(image_path)
            cache_file = self.segment_labels_cache_dir / f"{cache_key}.pkl"
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(labels, f)
            except:
                pass  # 캐시 저장 실패해도 계속 진행
        
        return labels
    
    def process_single_background(
        self,
        bg_path: Path,
        location_embeddings: torch.Tensor,
        semantic_locations: List[str]
    ) -> Dict:
        """
        단일 배경 이미지 처리
        
        Args:
            bg_path: 배경 이미지 경로
            location_embeddings: 의미론적 위치 임베딩
            semantic_locations: 의미론적 위치 텍스트 리스트
        
        Returns:
            매칭된 배경 정보 또는 None
        """
        try:
            print(f"Processing: {bg_path.name}")
            
            # 이미지 로드
            bg_img = Image.open(bg_path).convert("RGB")
            
            # Segmentation
            annotations, labeled_results = run_segmentation(bg_img)
            
            # 각 segment의 label을 임베딩하고 유사도 검사
            for idx, (mask, label) in enumerate(annotations):
                if mask.sum() == 0:
                    continue
                
                # segment label 임베딩 (캐싱 사용)
                segment_label = labeled_results[idx]["label"]
                segment_embedding = self.embedding_manager.embed_texts_with_cache([segment_label])
                
                # 각 의미론적 위치와 유사도 계산
                sim_list = []
                for loc_idx, loc_emb in enumerate(location_embeddings):
                    similarity = self.embedding_manager.compute_similarity(
                        segment_embedding, loc_emb.unsqueeze(0)
                    )
                    sim_list.append((similarity, loc_idx))
                    
                    if similarity >= self.similarity_threshold:
                        print(f"  ✓ Match found: '{segment_label}' ~ '{semantic_locations[loc_idx]}' (sim={similarity:.3f})")
                        
                        return {
                            "bg_path": bg_path,
                            "bg_image": bg_img,
                            "segment_mask": mask,
                            "segment_label": segment_label,
                            "matched_location": semantic_locations[loc_idx],
                            "similarity": similarity
                        }
            
            if sim_list:
                print(f"  Max Similarity: {max(sim_list)[0]:.3f}")
            return None
        
        except Exception as e:
            print(f"  ✗ Error processing {bg_path.name}: {e}")
            return None
    
    def find_suitable_backgrounds(
        self,
        semantic_locations: List[str],
        dataset_dir: Path,
        max_backgrounds: int = 5,
        max_workers: int = 5,
        broad_categories: List[str] = None
    ) -> List[Dict]:
        """
        적합한 배경 이미지 찾기 (병렬 처리)
        
        Args:
            semantic_locations: 의미론적 위치 리스트 (예: ["on table", "on desk"])
            dataset_dir: 데이터셋 디렉토리 경로
            max_backgrounds: 최대 반환할 배경 이미지 수
            max_workers: 병렬 처리 워커 수
            broad_categories: 대분류 카테고리 리스트 (예: ["home", "indoor"])
        
        Returns:
            적합한 배경 이미지와 segment 정보 리스트
        """
        # 이미지 확장자
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # 1. 의미론적 위치 텍스트 임베딩 (캐싱 사용)
        location_embeddings = self.embedding_manager.embed_texts_with_cache(semantic_locations)
        
        suitable_backgrounds = []
        
        # 2. 폴더 순서대로 이미지 처리
        if broad_categories:
            print(f"\n[Folder Filtering] Using broad categories: {broad_categories}")
            
            # 폴더 임베딩 계산 및 랭킹
            folder_embeddings = self.embedding_manager.get_folder_embeddings(dataset_dir)
            ranked_folders = self.embedding_manager.rank_folders_by_similarity(
                broad_categories, folder_embeddings
            )
            
            print(f"[Folder Filtering] Processing folders in similarity order...")
            
            # 유사도 높은 폴더부터 순차 처리
            for folder, similarity in ranked_folders:
                if len(suitable_backgrounds) >= max_backgrounds:
                    break
                
                # 폴더 내 이미지 파일 찾기 (on-demand)
                folder_images = []
                for file in folder.iterdir():
                    if file.is_file() and file.suffix.lower() in image_extensions:
                        folder_images.append(file)
                
                if not folder_images:
                    continue
                
                print(f"\n[Processing] {folder.name} ({similarity:.3f}): {len(folder_images)} images")
                
                # 이 폴더의 이미지들을 병렬 처리
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_bg = {
                        executor.submit(
                            self.process_single_background,
                            bg_path,
                            location_embeddings,
                            semantic_locations
                        ): bg_path
                        for bg_path in folder_images
                    }
                    
                    for future in as_completed(future_to_bg):
                        result = future.result()
                        if result is not None:
                            suitable_backgrounds.append(result)
                            
                            if len(suitable_backgrounds) >= max_backgrounds:
                                for f in future_to_bg:
                                    f.cancel()
                                break
        else:
            # 대분류 없으면 전체 디렉토리 스캔
            from utils import find_all_images
            
            print(f"\n[Processing] Scanning all images in {dataset_dir}...")
            all_images = find_all_images(dataset_dir, use_cache=True, cache_dir=self.cache_dir)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_bg = {
                    executor.submit(
                        self.process_single_background,
                        bg_path,
                        location_embeddings,
                        semantic_locations
                    ): bg_path
                    for bg_path in all_images
                }
                
                for future in as_completed(future_to_bg):
                    result = future.result()
                    if result is not None:
                        suitable_backgrounds.append(result)
                        
                        if len(suitable_backgrounds) >= max_backgrounds:
                            for f in future_to_bg:
                                f.cancel()
                            break
        
        return suitable_backgrounds
