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

// 프로세스 시작
window.start = async function() {
    const startSection = document.getElementById('start-section');
    startSection.innerHTML = '<h2>로딩 중...</h2><p>이미지와 기존 포즈를 불러옵니다.</p>';

    try {
        // 1. 서버에서 이미지 로드
        await loadImages();

        if (images.length === 0) {
            startSection.innerHTML = '<h2>오류</h2><p>서버에서 이미지를 찾을 수 없습니다.</p>';
            return;
        }

        // 2. 기존 포즈 로드 시도 (사용자에게 알림 없이)
        await loadExistingPoses(true); 

        // 3. UI 전환
        startSection.classList.add('hidden');
        document.getElementById('image-preview').classList.remove('hidden');
        document.getElementById('instructions').classList.remove('hidden');
        document.getElementById('controls').classList.remove('hidden');

        // 4. 프로세스 시작
        showCurrentImage();

    } catch (error) {
        console.error('시작 중 오류:', error);
        startSection.innerHTML = `<h2>오류</h2><p>시작 중 오류가 발생했습니다: ${error.message}</p>`;
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
        throw error; // Let the caller handle the error
    }
}

// 현재 이미지 표시
function showCurrentImage() {
    if (currentImageIndex >= images.length) {
        alert('모든 이미지 처리 완료! 저장 후 자동으로 다음 단계로 진행합니다.');
        savePoses(true);  // autoComplete = true
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

// 포즈 저장 (autoComplete=true면 확인 없이 자동 종료)
window.savePoses = async function(autoComplete = false) {
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
        
        // 서버 종료 (다음 단계로 진행)
        const shouldComplete = autoComplete || confirm('포즈 저장이 완료되었습니다. 서버를 종료하고 다음 단계로 진행하시겠습니까?');
        if (shouldComplete) {
            document.body.innerHTML = '<h1 style="text-align:center; margin-top:20vh; color:#fff;">✅ 포즈 저장 완료! 서버를 종료합니다...</h1>';
            // 서버 종료
            setTimeout(async () => {
                try {
                    await fetch(`${API_BASE}/api/complete`, { method: 'POST' });
                } catch (e) {
                    // 서버가 종료되면서 연결이 끊어지므로 에러 무시
                }
            }, 500);
        }
    } catch (error) {
        console.error('Save error:', error);
        alert('저장 실패: ' + error.message);
    }
}

// 기존 포즈 불러오기
window.loadExistingPoses = async function(silent = false) {
    try {
        const response = await fetch(`${API_BASE}/api/poses/load`);
        const data = await response.json();
        
        if (Object.keys(data.poses).length === 0) {
            if (!silent) {
                alert('저장된 포즈가 없습니다.');
            }
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
        
        if (!silent) {
            alert(`${Object.keys(poses).length}개의 포즈를 불러왔습니다!`);
        }
        
    } catch (error) {
        console.error('Load error:', error);
        if (!silent) {
            alert('불러오기 실패: ' + error.message);
        } else {
            // Re-throw if silent so the start() function can catch it
            throw error;
        }
    }
}

// 초기화
window.resetAll = async function() {
    if (!confirm('모든 데이터를 초기화하시겠습니까? (페이지가 새로고침됩니다)')) return;

    try {
        await fetch(`${API_BASE}/api/reset`, { method: 'DELETE' });
        alert('초기화 완료!');
        window.location.reload(); // Reload the page to go back to the start screen
    } catch (error) {
        console.error('Reset error:', error);
        alert('초기화 실패: ' + error.message);
    }
}

// 앱 시작
init();