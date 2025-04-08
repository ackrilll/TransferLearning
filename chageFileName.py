import os

file_path = r'.\archive\Bone Break Classification\Bone Break Classification\Test'

for class_name in os.listdir(file_path):
    class_path = os.path.join(file_path, class_name)
    file_names = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

    print(f"\n📂 클래스: {class_name} | 총 파일 수: {len(file_names)}")

    # 1단계: 임시 이름으로 변경
    for filename in file_names:
        old_path = os.path.join(class_path, filename)
        temp_path = os.path.join(class_path, f"_temp_{filename}")
        try:
            if not os.path.exists(temp_path):
                os.rename(old_path, temp_path)
        except FileNotFoundError:
            print(f"❌ 파일 없음: {old_path}")
            continue

    # 2단계: 최종 이름으로 변경
    temp_files = sorted([f for f in os.listdir(class_path) if f.startswith("_temp_")])
    for i, temp_name in enumerate(temp_files):
        temp_path = os.path.join(class_path, temp_name)
        new_path = os.path.join(class_path, f"{i}.jpg")
        try:
            os.rename(temp_path, new_path)
        except FileExistsError:
            print(f"⚠️ 이미 존재하는 파일 (스킵): {new_path}")
            continue

    print(f"✅ {class_name} 클래스 이름 정리 완료!")
