import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// 전역 변수
let scene, camera, renderer, controls;
let sphere;
let images = [];
let currentImageIndex = 0;
let poses = {};
let frustumHelpers = [];

const SPHERE_RADIUS = 5;
const API_BASE = window.location.origin;

// 초기화
function init() {
    // Scene 설정
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Camera 설정
    camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.set(0, 5, 10);
    camera.lookAt(0, 0, 0);

    // Renderer 설정
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // Orbit Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enablePan = false; // 팬 비활성화
    controls.target.set(0, 0, 0);
    controls.update();

    // 조명
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // 중심 구 (wireframe)
    const sphereGeometry = new THREE.SphereGeometry(SPHERE_RADIUS, 32, 32);
    const sphereMaterial = new THREE.MeshBasicMaterial({
        color: 0x00ff00,
        wireframe: true,
        transparent: true,
        opacity: 0.3
    });
    sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    scene.add(sphere);

    // 그리드 헬퍼
    const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    scene.add(gridHelper);

    // 축 헬퍼
    const axesHelper = new THREE.AxesHelper(8);
    scene.add(axesHelper);

    // 정면 표시 (Z+ 방향)
    const arrowHelper = new THREE.ArrowHelper(
        new THREE.Vector3(0, 0, 1), // 방향: Z+ (정면)
        new THREE.Vector3(0, 0, 0), // 시작점
        7, // 길이
        0xff0000, // 빨간색
        2, // 머리 길이
        1 // 머리 너비
    );
    scene.add(arrowHelper);

    // "FRONT" 텍스트 표시용 스프라이트
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ff0000';
    ctx.font = 'bold 60px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('FRONT', 128, 80);
    
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.position.set(0, 0, 8);
    sprite.scale.set(4, 2, 1);
    scene.add(sprite);

    // 이벤트 리스너
    window.addEventListener('resize', onWindowResize);
    document.addEventListener('keydown', onKeyDown);
    document.getElementById('file-input').addEventListener('change', handleFileUpload);

    // 애니메이션 시작
    animate();
}

// 애니메이션 루프
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// 윈도우 리사이즈
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// 키보드 이벤트
function onKeyDown(event) {
    if (event.code === 'Space' || event.code === 'Enter') {
        event.preventDefault();
        if (images.length > 0 && currentImageIndex < images.length) {
            captureCurrentPose();
        }
    }
}

// 파일 업로드 처리
async function handleFileUpload(event) {
    const files = event.target.files;
    console.log('Selected files:', files.length);
    
    if (files.length === 0) {
        alert('파일을 선택해주세요.');
        return;
    }

    // 로딩 표시
    const uploadSection = document.getElementById('upload-section');
    const originalHTML = uploadSection.innerHTML;
    uploadSection.innerHTML = '<h2>업로드 중...</h2><p>잠시만 기다려주세요.</p>';

    const formData = new FormData();
    for (let file of files) {
        console.log('Adding file:', file.name);
        formData.append('files', file);
    }

    try {
        console.log('Uploading to:', `${API_BASE}/api/upload`);
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Upload result:', result);

        // 이미지 목록 가져오기
        await loadImages();
        
        if (images.length === 0) {
            throw new Error('업로드된 이미지가 없습니다. 이미지 파일을 선택했는지 확인해주세요.');
        }
        
        // UI 업데이트
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('image-preview').classList.remove('hidden');
        document.getElementById('instructions').classList.remove('hidden');
        document.getElementById('controls').classList.remove('hidden');

        // 첫 이미지 표시
        showCurrentImage();
    } catch (error) {
        console.error('Upload error:', error);
        alert('업로드 실패: ' + error.message + '\n\n브라우저 콘솔(F12)을 확인하세요.');
        uploadSection.innerHTML = originalHTML;
        // 이벤트 리스너 재등록
        document.getElementById('file-input').addEventListener('change', handleFileUpload);
    }
}

// 이미지 목록 로드
async function loadImages() {
    try {
        console.log('Loading images from:', `${API_BASE}/api/images`);
        const response = await fetch(`${API_BASE}/api/images`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        images = data.images;
        currentImageIndex = 0;
        console.log('Loaded images:', images);
        
        if (images.length === 0) {
            console.warn('No images found in response');
        }
    } catch (error) {
        console.error('Failed to load images:', error);
        throw error;
    }
}

// 현재 이미지 표시
function showCurrentImage() {
    if (currentImageIndex >= images.length) {
        alert('모든 이미지 처리 완료!');
        savePoses();
        return;
    }

    const filename = images[currentImageIndex];
    const imgElement = document.getElementById('current-image');
    imgElement.src = `${API_BASE}/api/images/${filename}`;

    document.getElementById('current-filename').textContent = filename;
    document.getElementById('progress-text').textContent = 
        `${currentImageIndex + 1}/${images.length}`;
    
    const progressPercent = ((currentImageIndex + 1) / images.length) * 100;
    document.getElementById('progress-fill').style.width = `${progressPercent}%`;
}

// 현재 카메라 포즈 캡처
window.captureCurrentPose = function() {
    if (currentImageIndex >= images.length) return;

    const filename = images[currentImageIndex];
    
    // 카메라의 현재 위치와 방향 저장
    const position = camera.position.toArray();
    const target = controls.target.toArray();
    const up = camera.up.toArray();

    poses[filename] = {
        position: position,
        look_at: target,
        up: up
    };

    console.log(`Pose captured for ${filename}:`, poses[filename]);

    // Frustum 헬퍼 추가
    addCameraFrustum(position, target);

    // 다음 이미지로
    currentImageIndex++;
    showCurrentImage();
}

// 카메라 frustum 시각화 추가
function addCameraFrustum(position, target) {
    // 작은 카메라 아이콘 생성
    const cameraHelper = new THREE.CameraHelper(
        new THREE.PerspectiveCamera(60, 1, 0.1, 2)
    );
    
    // 위치 설정
    const group = new THREE.Group();
    group.position.fromArray(position);
    group.lookAt(new THREE.Vector3().fromArray(target));
    
    // Frustum 시각화
    const frustumGeometry = new THREE.ConeGeometry(0.3, 1, 4);
    const frustumMaterial = new THREE.MeshBasicMaterial({
        color: 0xff6b6b,
        wireframe: true
    });
    const frustum = new THREE.Mesh(frustumGeometry, frustumMaterial);
    frustum.rotation.x = Math.PI / 2;
    frustum.position.z = -0.5;
    
    group.add(frustum);
    
    // 작은 구 (카메라 위치 표시)
    const sphereGeometry = new THREE.SphereGeometry(0.2, 8, 8);
    const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });
    const marker = new THREE.Mesh(sphereGeometry, sphereMaterial);
    group.add(marker);
    
    scene.add(group);
    frustumHelpers.push(group);
}

// 이미지 건너뛰기
window.skipImage = function() {
    currentImageIndex++;
    showCurrentImage();
}

// 포즈 저장
window.savePoses = async function() {
    try {
        const response = await fetch(`${API_BASE}/api/poses/save`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ poses: poses })
        });

        const result = await response.json();
        console.log('Save result:', result);
        alert(`${result.saved}개의 포즈가 저장되었습니다!`);

        // 다운로드
        window.open(`${API_BASE}/api/poses/download`, '_blank');
    } catch (error) {
        console.error('Save error:', error);
        alert('저장 실패: ' + error.message);
    }
}

// 기존 포즈 불러오기
window.loadExistingPoses = async function() {
    try {
        const response = await fetch(`${API_BASE}/api/poses/load`);
        const data = await response.json();
        
        if (Object.keys(data.poses).length === 0) {
            alert('저장된 포즈가 없습니다.');
            return;
        }

        poses = data.poses;
        
        // 기존 frustum 제거
        frustumHelpers.forEach(helper => scene.remove(helper));
        frustumHelpers = [];

        // Frustum 시각화
        for (const [filename, pose] of Object.entries(poses)) {
            addCameraFrustum(pose.position, pose.look_at);
        }

        alert(`${Object.keys(poses).length}개의 포즈를 불러왔습니다!`);
        
        // UI 업데이트
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('instructions').classList.remove('hidden');
        document.getElementById('controls').classList.remove('hidden');
        
    } catch (error) {
        console.error('Load error:', error);
        alert('불러오기 실패: ' + error.message);
    }
}

// 초기화
window.resetAll = async function() {
    if (!confirm('모든 데이터를 초기화하시겠습니까?')) return;

    try {
        await fetch(`${API_BASE}/api/reset`, { method: 'DELETE' });
        
        // 클라이언트 상태 초기화
        images = [];
        currentImageIndex = 0;
        poses = {};
        
        frustumHelpers.forEach(helper => scene.remove(helper));
        frustumHelpers = [];

        // UI 초기화
        document.getElementById('upload-section').classList.remove('hidden');
        document.getElementById('image-preview').classList.add('hidden');
        document.getElementById('instructions').classList.add('hidden');
        document.getElementById('controls').classList.add('hidden');

        alert('초기화 완료!');
    } catch (error) {
        console.error('Reset error:', error);
        alert('초기화 실패: ' + error.message);
    }
}

// 앱 시작
init();
