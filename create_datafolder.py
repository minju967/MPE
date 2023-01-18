from datetime import datetime
import shutil
import argparse
import random
import pandas as pd
import glob
import os


def Separate_image(opt):
    label_df = pd.read_csv(opt.csv)

    for idx in range(len(label_df.index)):
        row         = label_df.iloc[idx]        # 행번호로 불러오기
        img_name    = row['Name']
        img_path    = row['Path']
        _labeled    = row['Labeled']
        img_label   = list(row['A':'None'].to_list())
        classes     = ['A', 'HSM', 'C', 'Cut', 'Wire', 'None']
        
        unlabeled_path = os.path.join('D:\\VS_CODE\\Paper\\Data', 'unlabeled')
        os.makedirs(unlabeled_path, exist_ok=True)
        un_dst_path = os.path.join(unlabeled_path, img_name)

        if _labeled:
            for i in range(6):
                cls_path = os.path.join('D:\\VS_CODE\\Paper\\Data', classes[i])
                os.makedirs(cls_path, exist_ok=True)
                dst_path = os.path.join(cls_path, img_name)

                if img_label[i]:
                    shutil.copyfile(src=img_path, dst=dst_path)
        else:
            shutil.copyfile(src=img_path, dst=un_dst_path)

    return

def create_datafolder(opt):
    folders = os.listdir(opt.save)  # Data folder에 들어있는 파일 리스트
    num_list = []
    for f in folders:
        if 'dataset_' in f:
            num = f.split('_')[-1]
            num_list.append(num)

    num_list.sort()
    if len(num_list) == 0:
        folder_name = 'dataset_0'
    else:    
        folder_name = f'dataset_{int(num_list[-1])+1}'
    
    os.makedirs(os.path.join(opt.save, folder_name, 'Train'))
    os.makedirs(os.path.join(opt.save, folder_name, 'Test'))

    return os.path.join(opt.save, folder_name)

def create_dataset(opt):

    random.seed(datetime.now().timestamp())

    save_df_path = create_datafolder(opt)
    All_df = pd.read_csv(opt.csv)
    train_df = pd.DataFrame(index=range(0,0), columns=['Idx', 'Name', 'A', 'HSM', 'C', 'Cut', 'Wire', 'None', 'Path', 'Labeled'])
    test_df = pd.DataFrame(index=range(0,0), columns=['Idx', 'Name', 'A', 'HSM', 'C', 'Cut', 'Wire', 'None', 'Path', 'Labeled'])

    # class_list = os.listdir(opt.path)
    class_list = ['A', 'HSM', 'C', 'Cut', 'Wire']
    class_list.sort()

    train_data = []
    test_data  = []

    for cls in class_list:
        cls_path = os.path.join(opt.path, cls)
        num = len(glob.glob(cls_path+'\\*.png'))
        num_of_test = int(num * 0.2)
        idx_list = list(range(num))
         
        # Test Image 선택
        print(f'\n{cls}_Test Image: {num_of_test}')
        for idx in range(num_of_test):
            file_name = glob.glob(cls_path+'\\*.png')[idx].split('\\')[-1]
            test_data.append(glob.glob(cls_path+'\\*.png')[idx])
            idx_list.remove(idx)
            test_df = pd.concat([test_df, All_df.loc[All_df.Name==file_name]], ignore_index=True, bool_only=True)
        
        # Train Image 선택
        print(f'{cls}_Train Image: {len(idx_list)}')
        for idx in idx_list:
            file_name = glob.glob(cls_path+'\\*.png')[idx].split('\\')[-1]
            train_data.append(glob.glob(cls_path+'\\*.png')[idx])
            train_df = pd.concat([train_df, All_df.loc[All_df.Name==file_name]], ignore_index=True, bool_only=True)

    # 중복제거
    train_data = list(set(train_data))
    test_data = list(set(test_data))
    
    # dataset_{}\\Train\\*.png
    for src in train_data:
        # src: image path
        file_name = src.split('\\')[-1]
        dst = os.path.join(save_df_path, 'Train', file_name)
        shutil.copyfile(src, dst)
    train_df.to_csv(os.path.join(save_df_path, 'Train', 'label.csv'))

    # dataset_{}\\Test\\*.png
    for src in test_data:
        # src: image path
        file_name = src.split('\\')[-1]
        dst = os.path.join(save_df_path, 'Test', file_name)
        shutil.copyfile(src, dst)
    test_df.to_csv(os.path.join(save_df_path, 'Test', 'label.csv'))

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='D:\\VS_CODE\\Paper\\Data', help="folder_name")                     # Data folder에는 클래스별 데이터
    parser.add_argument('--save', type=str, default='D:\\VS_CODE\\Paper\\Datafolder', help="Save folder")               # Datafolder folder에는 학습/평가에 사용될 데이터셋 폴더
    parser.add_argument('--csv', type=str, default='D:\\VS_CODE\\Paper\\Data\\labeled_data.csv', help="data_label.csv")
    args = parser.parse_args()
    
    # Separate_image(args)
    create_dataset(args)
    