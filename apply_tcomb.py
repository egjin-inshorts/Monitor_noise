import vapoursynth as vs

core = vs.core
core.num_threads = 10

source_path = "/Users/easyjin/Downloads/SD_Thestepsofheaven_12th_ss004542t000100.mov"
mode = 2      # 0: luma, 1: chroma, 2: luma+chroma
fthreshl = 4  # 밝기 프레임 내 변화 감지 임계값  (낮을 수록 민감하게 잔상 감지하고 제거)
fthreshc = 5  # 색상 프레임 내 변화 감지 임계값
othreshl = 5  # 밝기 프레임 간 변화 감지 임계값
othreshc = 6  # 색상 프레임 간 변화 감지 임계값
scthresh = 12.0

src = core.ffms2.Source(source=source_path)

W = src.width
H = src.height
print(f"원본 해상도: {W}x{H}")

bits = src.format.bits_per_sample
print(f"원본 비트 심도: {bits}")

# 패딩 크기 계산 (TComb 필터가 필요로 하는 포맷 요구사항 충족)
pad_right = 1 if (W % 2 != 0) else 0
pad_bottom = (4 - (H % 4)) if (H % 4 != 0) else 0

if pad_right > 0 or pad_bottom > 0:
    src = core.std.AddBorders(
        src,
        left=0,
        right=pad_right,
        top=0,
        bottom=pad_bottom,
        color=[0, 0, 0]
    )
    print(f"패딩 후 해상도: {src.width}x{src.height}")

down8 = core.resize.Bicubic(src, format=vs.YUV422P8)

tc_processed_8 = core.tcomb.TComb(
    down8, 
    mode=mode, 
    fthreshl=fthreshl,
    fthreshc=fthreshc,
    othreshl=othreshl,
    othreshc=othreshc,
    map=0,
    scthresh=scthresh
)
tc_mask_8 = core.tcomb.TComb(
    down8, 
    mode=mode,
    fthreshl=fthreshl,
    fthreshc=fthreshc,
    othreshl=othreshl,
    othreshc=othreshc,
    map=1,
    scthresh=scthresh
)

tc_processed = core.resize.Bicubic(tc_processed_8, format=src.format.id)
tc_mask = core.resize.Bicubic(tc_mask_8, format=src.format.id)

dst = core.std.MaskedMerge(src, tc_processed, tc_mask)

if pad_right > 0 or pad_bottom > 0:
    dst = core.std.Crop(dst, right=pad_right, bottom=pad_bottom)
    assert dst.width == W and dst.height == H, f"[ERROR] 크롭 후 크기({dst.width}x{dst.height})가 원본 크기({W}x{H})와 일치하지 않습니다!"

dst.set_output()
