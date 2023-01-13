# MPE

Create_datafolder.py
    
    <데이터셋 생성 파일>

    '''
        DATASET
        |
        |--- dataset_1
            |
            |--- Train
            |      |
            |      |--- Images
            |      |        |
            |      |        |---img1.png
            |      |        |---img2.png
            |      |        |---img3.png
            |      |--- label.csv
            |--- Test
                    |
                    |--- Images
                    |        |
                    |        |---img1.png
                    |        |---img2.png
                    |        |---img3.png
                    |--- label.csv
    '''

    클래스: 연삭전극, 고속가공, 고속전극, 연삭, 와이어, None 
    label:  A, HSP, C, Grinding, Wire, None

    클래스간의 데이터 imbalance가 크기 때문에 Random하게 train/test를 split하게 되면 비율이 맞지 않다.
    테스트 데이터에 없는 클래스가 있을 수 있다.
    따라서, 각 클래스 별 데이터를 8:2 비율로 train/test로 분리된다.

    데이터의 label은 Multi-label을 가지고 있기 때문에 test로 분리된 이미지는 뒤에 클래스에서 다시 선택될 수 있다.
    따라서 train / test 데이터 비율이 8:2가 안될 수도 있다. 

Separate_image

    <이미지 클래스 분류>

    Create_datafolder.py를 실행하기 위한 base datafolder 생성