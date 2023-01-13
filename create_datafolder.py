from datetime import datetime
import shutil
import argparse
import random
import pandas as pd
import glob
import os



def Separate_image(opt):
    # label_df = pd.read_csv(opt.csv)


    label_df = pd.DataFrame(data=[[0, 'A123.png', 0,1,1,0,0,1,'C:\\home', True]], columns=['Idx', 'Name', 'A', 'HSM', 'C', 'Grinding', 'Wire', 'None', 'Path', 'Labeled'])
    unlabeled_path = None

    for idx in range(len(label_df.index)):
        row = label_df.iloc[idx]        # 행번호로 불러오기
        img_path = row['Path']
        img_label = list(row['A':'None'].to_list())
        _labeled = row['Labeled']

        if _labeled:
            for i in range(6):
                cls_path = None
                if img_label[i]:
                    pass
                else:
                    shutil.copyfile(src=img_path, dst=cls_path)
                    pass
        else:
            shutil.copyfile(src=img_path, dst=unlabeled_path)

    return

def create_datafolder(path):
    folders = os.listdir(path)  # Data folder에 들어있는 파일 리스트
    num_list = []
    for f in folders:
        if 'dataset_' in f:
            num = f.split('_')[-1]
            num_list.append(num)

    num_list.sort()
    if len(num_list) == 0:
        folder_name = 'dataset_0'
    else:    
        folder_name = f'dataset_{num_list[-1]+1}'
    
    os.makedirs(os.path.join(path, folder_name, 'Train'))
    os.makedirs(os.path.join(path, folder_name, 'Test'))

    return os.path.join(path, folder_name)

def create_dataset(opt):

    random.seed(datetime.now().timestamp())

    save_df_path = create_datafolder(opt.save)
    All_df = pd.read_csv(opt.csv)
    train_df = pd.DataFrame(index=range(0,0), columns=['Idx', 'Name', 'A', 'HSM', 'C', 'Grinding', 'Wire', 'None', 'Path', 'Labeled'])
    test_df = pd.DataFrame(index=range(0,0), columns=['Idx', 'Name', 'A', 'HSM', 'C', 'Grinding', 'Wire', 'None', 'Path', 'Labeled'])

    # class_list = os.listdir(opt.path)
    class_list = ['A', 'B', 'C', 'D', 'E']
    class_list.sort()

    train_data = []
    test_data  = []

    for cls in class_list:
        cls_path = os.path.join(opt.path, cls)
        num = len(glob.glob(cls_path+'\\*.png'))
        num_of_test = int(num * 0.2)

        idx_list = list(range(num))
        
        # Test Image 선택
        for idx in num_of_test:
            file_name = glob.glob(cls_path+'\\*.png')[idx].split('\\')[-1]
            test_data.append(glob.glob(cls_path+'\\*.png')[idx])
            idx_list.remove(idx)
            test_df.loc[len(test_df)] = All_df[All_df['Name'] == file_name]
        
        # Train Image 선택
        for idx in idx_list:
            file_name = glob.glob(cls_path+'\\*.png')[idx].split('\\')[-1]
            train_data.append(glob.glob(cls_path+'\\*.png')[idx])
            train_df.loc[len(train_df)] = All_df[All_df['Name'] == file_name]

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
    parser.add_argument('--csv', type=str, default='D:\\VS_CODE\\Paper\\Data\\label.csv', help="data_label.csv")
    args = parser.parse_args()
    
    Separate_image(args.csv)
    # create_dataset(args.path)
    