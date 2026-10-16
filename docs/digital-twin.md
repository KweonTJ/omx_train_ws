# OMX Digital Twin

[Lab에서 선택](https://kweontj.github.io/KweonTJ/lab/) · [웹 선반 트윈](https://kweontj.github.io/KweonTJ/simulator/)

웹 트윈과 네이티브 MuJoCo의 색상, 조명, 그리드, 초기 시점을 맞췄다. 기존 학습용 MJCF, mesh, manifest, 정책과 환경 코드는 변경하지 않는다. 디자인은 뷰어 메모리에만 적용한다.

## 실행

화면 출력이 가능한 PC, Python 3.11 이상이 필요하다. GPU 학습 의존성 없이 별도 환경에서 실행할 수 있다.

```bash
git clone https://github.com/KweonTJ/omx_train_ws.git
cd omx_train_ws
python -m venv .venv-twin
source .venv-twin/bin/activate  # Windows: .venv-twin\Scripts\activate
python -m pip install mujoco numpy

# 최신 선반 표시 장면 (기본)
python scripts/view_twin.py --scene shelf
# 기존 배송 물리 샌드박스
python scripts/view_twin.py --scene delivery
```

MuJoCo 오른쪽 **Control** 패널에서 joint1–4와 그리퍼 목표를 조절한다. 패널이 접혀 있으면 제목을 눌러 펼친다. `R`로 초기화, 드래그로 회전, 휠로 확대한다. macOS의 passive viewer는 `python` 대신 MuJoCo 패키지가 제공하는 `mjpython`으로 실행한다.

| 장면 | 동작 | 정책 |
|---|---|---|
| `shelf` | 웹과 같은 트랙형 리더·100 mm 받침·2단 선반·ArUco 1–4, `mj_forward` 기구학 표시 | 실행하지 않음 |
| `delivery` | 원래 타워·상자와 `mj_step` 물리, 기존 8/8/5 계약, 수동 관절 목표 | 실행하지 않음 |

선반 장면은 시각적 검토용이다. 질량·관성은 표시 모델 컴파일을 위한 임시 값이며, 충돌·중력·접촉·파지를 검증하는 물리 모델이 아니다. 기존 배송 PPO를 새 선반 정책으로 사용하지 않는다. ROS 2 연결은 웹 선반 트윈의 수신 기능에서 제공한다. 이 네이티브 뷰어는 하드웨어 명령을 발행하지 않는다.

기존 **PPO 재생 뷰어**에도 같은 스타일을 적용했다. 학습 환경 의존성을 갖춘 원래 환경에서 아래처럼 최종 배포 artifact를 지정한다. 새 선반 표시 장면은 이 경로에 주입하지 않는다.

```bash
uv run --frozen python scripts/view_grasp_policy.py \
  --policy policies/exported/arm_delivery_pitch_hold_v4/policy.zip
```

## 모델 출처와 재생성

`assets/twin/source.json`은 웹 트윈의 동일 snapshot이다. 원본은 `KweonTJ/Omx_Turtle_Bot` commit `3bd5eaa6a74788ac53997f5d826e23c35e49770b`의 리더 URDF와 `export_raised_shelf_scene.py`다. 파일별 SHA-256과 ArUco 패턴을 포함하고, Apache 2.0 원본 라이선스는 `assets/twin/UPSTREAM-LICENSE`에 보관한다. STL과 보드 텍스처를 수정하지 않았다.

```bash
python scripts/export_shelf_twin.py
python scripts/test_load_model.py
python -m pytest test/test_digital_twin.py -q
```

생성기는 모든 원본 asset의 해시를 검증하고 `assets/twin/shelf.xml`만 재생성한다. 원본 URDF의 고정축 RPY를 MJCF quaternion으로 변환한다. 테스트는 세 자세에서 모든 body의 위치·회전을 원본 FK와 비교하고, 그리퍼 mimic·범위·초기화와 기존 모델의 1,000-step 물리 결과가 스타일 적용 전후 동일한지 확인한다.

검증 환경: Python 3.12, MuJoCo 3.3.7, EGL. 위 native 테스트 5개와 원래 모델 manifest/1,000-step 검사가 통과했고 두 장면의 실제 렌더링을 확인했다. 클라우드에 화면 서버가 없어 데스크톱 GUI의 마우스·키보드 수동 조작과 전체 PPO 재생은 실행하지 않았다.
