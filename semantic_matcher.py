"""
Semantic background matching module
의미론적으로 적합한 배경 위치를 찾는 모듈
"""

from pathlib import Path
from PIL import Image
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import cv2
import numpy as np
from tqdm import tqdm

from segment import run_segmentation
from embedding import EmbeddingManager
from utils import get_image_cache_key, find_all_images
import pickle


class SemanticMatcher:
    """의미론적 배경 매칭 클래스"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        cache_dir: Path = None,
        upscale_backgrounds: bool = True,
        upscale_factor: float = 4.0
    ):
        """
        Args:
            similarity_threshold: 유사도 임계값
            cache_dir: 캐시 디렉토리
            upscale_backgrounds: 배경 이미지 업스케일링 여부
            upscale_factor: 업스케일 배율 (기본 4배)
        """
        self.similarity_threshold = similarity_threshold
        self.upscale_backgrounds = upscale_backgrounds
        self.upscale_factor = upscale_factor
        
        # 캐시 설정
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".dataset_cache"
        cache_dir.mkdir(exist_ok=True)
        
        self.cache_dir = cache_dir
        self.segment_labels_cache_dir = cache_dir / "segment_labels"
        self.segment_labels_cache_dir.mkdir(exist_ok=True)
        
        # Segment 매칭 결과 캐시
        self.segment_match_cache = {}  # {(bg_path, locations_tuple): match_result}
        self.segment_match_cache_file = cache_dir / "segment_match_cache.pkl"
        self._load_segment_match_cache()
        
        # 임베딩 매니저 초기화
        self.embedding_manager = EmbeddingManager(cache_dir=cache_dir)
    
    def _load_segment_match_cache(self):
        """Segment 매칭 결과 캐시 로드"""
        if self.segment_match_cache_file.exists():
            try:
                with open(self.segment_match_cache_file, 'rb') as f:
                    self.segment_match_cache = pickle.load(f)
                print(f"  [Cache] Loaded {len(self.segment_match_cache)} segment match results")
            except:
                self.segment_match_cache = {}
    
    def _save_segment_match_cache(self):
        """Segment 매칭 결과 캐시 저장"""
        try:
            with open(self.segment_match_cache_file, 'wb') as f:
                pickle.dump(self.segment_match_cache, f)
        except:
            pass  # 저장 실패해도 계속 진행
    
    def upscale_image(self, img: Image.Image) -> Image.Image:
        """
        이미지를 고해상도로 업스케일링
        
        Args:
            img: PIL Image 객체
        
        Returns:
            업스케일된 PIL Image
        """
        if not self.upscale_backgrounds or self.upscale_factor <= 1.0:
            return img
        
        w, h = img.size
        new_w = int(w * self.upscale_factor)
        new_h = int(h * self.upscale_factor)
        
        # OpenCV로 변환하여 LANCZOS4 업스케일링
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        upscaled_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        upscaled_rgb = cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB)
        
        # print(f"    [Upscale] {w}x{h} → {new_w}x{new_h} ({self.upscale_factor}x)")
        
        return Image.fromarray(upscaled_rgb)
    
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
        단일 배경 이미지 처리 (캐싱 지원)
        
        Args:
            bg_path: 배경 이미지 경로
            location_embeddings: 의미론적 위치 임베딩
            semantic_locations: 의미론적 위치 텍스트 리스트
        
        Returns:
            매칭된 배경 정보 또는 None
        """
        # 캐시 키 생성 (경로 + locations)
        cache_key = (str(bg_path), tuple(semantic_locations))
        
        # 캐시에서 확인
        if cache_key in self.segment_match_cache:
            cached_result = self.segment_match_cache[cache_key]
            if cached_result is None:
                return None  # 이전에 매칭 실패한 배경
            
            # 캐시된 결과 반환 (이미지만 다시 로드)
            try:
                bg_img = Image.open(bg_path).convert("RGB")
                bg_img = self.upscale_image(bg_img)
                
                return {
                    "bg_path": bg_path,
                    "bg_image": bg_img,
                    "segment_mask": cached_result["segment_mask"],
                    "segment_label": cached_result["segment_label"],
                    "matched_location": cached_result["matched_location"],
                    "similarity": cached_result["similarity"]
                }
            except:
                # 캐시 무효화
                del self.segment_match_cache[cache_key]
        
        try:
            # 이미지 로드
            bg_img = Image.open(bg_path).convert("RGB")
            
            # 업스케일링 (설정된 경우)
            bg_img = self.upscale_image(bg_img)
            
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
                        # 매칭 성공 - 캐시에 저장
                        match_result = {
                            "segment_mask": mask,
                            "segment_label": segment_label,
                            "matched_location": semantic_locations[loc_idx],
                            "similarity": similarity
                        }
                        self.segment_match_cache[cache_key] = match_result
                        
                        return {
                            "bg_path": bg_path,
                            "bg_image": bg_img,
                            **match_result
                        }
            
            # 매칭 실패 - 캐시에 None 저장
            self.segment_match_cache[cache_key] = None
            return None
        
        except Exception as e:
            # 에러 발생 - 캐시하지 않음 (다음에 재시도)
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
            
            # 다양성을 위해 각 폴더에서 최대 개수 제한
            max_per_folder = max(200, max_backgrounds // 4)  # 최소 200개, 또는 전체의 1/4
            print(f"[Folder Filtering] Max per folder: {max_per_folder}")
            
            # 유사도 높은 폴더부터 순차 처리
            pbar = tqdm(total=max_backgrounds, desc="Finding backgrounds", unit="bg")
            folder_counts = {}  # 각 폴더에서 뽑은 개수 추적
            
            try:
                for folder, similarity in ranked_folders:
                    if len(suitable_backgrounds) >= max_backgrounds:
                        break
                    
                    # 이 폴더에서 뽑을 수 있는 최대 개수
                    remaining_for_folder = max_per_folder - folder_counts.get(folder.name, 0)
                    if remaining_for_folder <= 0:
                        continue  # 이 폴더는 이미 충분히 뽑음
                    
                    # 폴더 내 이미지 파일 찾기 (on-demand)
                    folder_images = []
                    for file in folder.iterdir():
                        if file.is_file() and file.suffix.lower() in image_extensions:
                            folder_images.append(file)
                    
                    if not folder_images:
                        continue
                    
                    pbar.set_description(f"Scanning {folder.name}")
                    folder_counts[folder.name] = folder_counts.get(folder.name, 0)
                    
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
                        
                        processed_count = 0  # 처리된 이미지 수 (매칭 성공/실패 포함)
                        
                        for future in as_completed(future_to_bg):
                            try:
                                result = future.result(timeout=10)  # 10초 타임아웃
                                processed_count += 1
                                
                                if result is not None:
                                    suitable_backgrounds.append(result)
                                    folder_counts[folder.name] += 1
                                    pbar.update(1)
                                    pbar.set_postfix({
                                        "folder": folder.name[:20], 
                                        "sim": f"{similarity:.2f}",
                                        "count": f"{folder_counts[folder.name]}/{max_per_folder}"
                                    })
                                    
                                    # 전체 목표 달성 또는 이 폴더 할당량 달성
                                    if len(suitable_backgrounds) >= max_backgrounds or \
                                       folder_counts[folder.name] >= max_per_folder:
                                        for f in future_to_bg:
                                            f.cancel()
                                        break
                                
                                # 매칭 실패가 너무 많으면 폴더 스킵 (효율성)
                                # 50개 처리했는데 5개도 안 매칭되면 스킵 (10% 미만)
                                if processed_count >= 50 and folder_counts[folder.name] < 5:
                                    print(f"\n  Skipping {folder.name}: low match rate ({folder_counts[folder.name]}/{processed_count})")
                                    for f in future_to_bg:
                                        f.cancel()
                                    break
                                    
                            except Exception as e:
                                # 타임아웃 또는 에러 발생 시 스킵
                                continue
                
            except KeyboardInterrupt:
                print(f"\n\n[Interrupted] User stopped the process. Found {len(suitable_backgrounds)}/{max_backgrounds} backgrounds.")
                print("[Interrupted] Saving cache and proceeding with partial results...\n")
                # 캐시 저장
                self._save_segment_match_cache()
            finally:
                pbar.close()
        else:
            # 대분류 없으면 전체 디렉토리 스캔
            
            print(f"\n[Processing] Scanning all images in {dataset_dir}...")
            all_images = find_all_images(dataset_dir, use_cache=True, cache_dir=self.cache_dir)
            
            try:
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
                                
            except KeyboardInterrupt:
                print(f"\n\n[Interrupted] User stopped the process. Found {len(suitable_backgrounds)}/{max_backgrounds} backgrounds.")
                print("[Interrupted] Saving cache and proceeding with partial results...\n")
                # 캐시 저장
                self._save_segment_match_cache()
        
        # 캐시 저장
        self._save_segment_match_cache()
        print(f"\n  [Cache] Saved {len(self.segment_match_cache)} segment match results")
        
        return suitable_backgrounds

