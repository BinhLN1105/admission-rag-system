import os
import shutil
from datetime import datetime

def backup_models():
    """Backup các mô hình hiện tại trước khi retrain"""

    base_dir = os.getcwd()  # Sử dụng thư mục hiện tại thay vì dựa vào __file__
    models_dir = os.path.join(base_dir, "models")
    backup_dir = os.path.join(base_dir, "models_backup")

    # Tạo thư mục backup nếu chưa có
    os.makedirs(backup_dir, exist_ok=True)

    # Tạo tên thư mục backup với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subdir = os.path.join(backup_dir, f"backup_{timestamp}")

    print(f"📦 Đang backup mô hình vào: {backup_subdir}")

    # Copy các file mô hình
    model_files = [
        "logistic_regression.pkl",
        "random_forest.pkl",
        "scaler.pkl"
    ]

    copied_files = []
    for file in model_files:
        src = os.path.join(models_dir, file)
        if os.path.exists(src):
            # Tạo thư mục đích
            os.makedirs(backup_subdir, exist_ok=True)
            dst = os.path.join(backup_subdir, file)
            shutil.copy2(src, dst)
            copied_files.append(file)
            print(f"  ✅ {file}")

    if copied_files:
        print(f"\n💾 Đã backup {len(copied_files)} file mô hình")
        print(f"📁 Đường dẫn: {backup_subdir}")

        # Tạo file info về backup
        info_file = os.path.join(backup_subdir, "backup_info.txt")
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"Backup created: {datetime.now()}\n")
            f.write(f"Files backed up: {', '.join(copied_files)}\n")
            f.write(f"Original location: {models_dir}\n")

        return backup_subdir
    else:
        print("⚠ Không tìm thấy file mô hình nào để backup")
        return None

def list_backups():
    """Liệt kê các backup có sẵn"""

    base_dir = os.getcwd()
    backup_dir = os.path.join(base_dir, "models_backup")

    if not os.path.exists(backup_dir):
        print("📁 Chưa có backup nào")
        return

    print("📋 DANH SÁCH BACKUP:")
    print("-" * 50)

    backups = []
    for item in os.listdir(backup_dir):
        item_path = os.path.join(backup_dir, item)
        if os.path.isdir(item_path):
            backups.append(item)

    if not backups:
        print("📁 Chưa có backup nào")
        return

    # Sắp xếp theo thời gian (mới nhất trước)
    backups.sort(reverse=True)

    for backup in backups:
        backup_path = os.path.join(backup_dir, backup)
        info_file = os.path.join(backup_path, "backup_info.txt")

        # Đọc thông tin backup
        info = f"Thư mục: {backup}"
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Backup created:"):
                            info += f" | {line.strip()}"
                            break
            except:
                pass

        print(f"  📦 {info}")

        # Kiểm tra file trong backup
        files = [f for f in os.listdir(backup_path) if f.endswith('.pkl') or f.endswith('.txt')]
        if files:
            print(f"      Files: {', '.join(files)}")

def restore_backup(backup_name):
    """Khôi phục từ backup"""

    base_dir = os.getcwd()
    models_dir = os.path.join(base_dir, "models")
    backup_dir = os.path.join(base_dir, "models_backup")
    backup_path = os.path.join(backup_dir, backup_name)

    if not os.path.exists(backup_path):
        print(f"❌ Không tìm thấy backup: {backup_name}")
        return False

    print(f"🔄 Đang khôi phục từ: {backup_path}")

    # Copy các file từ backup
    model_files = [
        "logistic_regression.pkl",
        "random_forest.pkl",
        "scaler.pkl"
    ]

    restored_files = []
    for file in model_files:
        src = os.path.join(backup_path, file)
        if os.path.exists(src):
            dst = os.path.join(models_dir, file)
            shutil.copy2(src, dst)
            restored_files.append(file)
            print(f"  ✅ {file}")

    if restored_files:
        print(f"\n🔄 Đã khôi phục {len(restored_files)} file mô hình")
        return True
    else:
        print("⚠ Không tìm thấy file mô hình nào trong backup")
        return False

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("📖 Cách sử dụng:")
        print("  python backup_models.py backup     # Tạo backup")
        print("  python backup_models.py list       # Xem danh sách backup")
        print("  python backup_models.py restore <tên_backup>  # Khôi phục")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "backup":
        backup_models()
    elif command == "list":
        list_backups()
    elif command == "restore":
        if len(sys.argv) < 3:
            print("❌ Cần chỉ định tên backup để khôi phục")
            print("📋 Xem danh sách: python backup_models.py list")
        else:
            backup_name = sys.argv[2]
            success = restore_backup(backup_name)
            if success:
                print("\n✅ Khôi phục thành công!")
                print("🔄 Khởi động lại server để áp dụng:")
                print("   python app/main.py")
            else:
                print("\n❌ Khôi phục thất bại!")
    else:
        print(f"❌ Lệnh không hợp lệ: {command}")
        print("📖 Lệnh hợp lệ: backup, list, restore")