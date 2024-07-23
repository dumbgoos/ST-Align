import os
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_file(file_path):
    """
    尝试读取 .pkl 文件并遍历其内容，记录不能正常读取的 data loaders，返回读取状态和文件路径。
    :param file_path: .pkl 文件路径
    :return: (文件路径, 是否能正常读取, 失败的 data loaders)
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        failed_loaders = []
        for item in data:
            try:
                # Assuming each item represents a data loader or similar object
                # and needs to be processed in a specific way
                # Replace the following with actual processing logic as needed
                _ = item  # Process item
            except Exception as e:
                failed_loaders.append(item)
        return (file_path, True, failed_loaders)
    except Exception as e:
        return (file_path, False, [])

def check_pkl_files(directory, max_workers=30):
    """
    检查指定目录下的所有 .pkl 文件，并记录不能正常读取的文件和 data loaders。
    :param directory: 目标目录路径
    :param max_workers: 最大并行线程数
    """
    invalid_files = []
    failed_loaders_dict = {}
    pkl_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_file, file): file for file in pkl_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking .pkl files"):
            file_path, is_valid, failed_loaders = future.result()
            if not is_valid or failed_loaders:
                invalid_files.append(file_path)
                if failed_loaders:
                    failed_loaders_dict[file_path] = failed_loaders

    return invalid_files, failed_loaders_dict

def main():
    directory = '/mnt/public/luoling/FoundaST/code/dataloader/valid'
    invalid_files, failed_loaders_dict = check_pkl_files(directory)

    if invalid_files:
        print("The following files could not be read or contain failed data loaders:")
        for file_path in invalid_files:
            print(file_path)
            if file_path in failed_loaders_dict:
                print(f"  Failed data loaders in {file_path}:")
                for loader in failed_loaders_dict[file_path]:
                    print(f"    {loader}")
    else:
        print("All .pkl files were read and processed successfully.")

if __name__ == "__main__":
    main()

"""
/mnt/public/luoling/FoundaST/code/dataloader/train/train_loader503.pkl
/mnt/public/luoling/FoundaST/code/dataloader/train/train_loader505.pkl
"""
