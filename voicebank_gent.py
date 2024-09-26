import os

from sklearn import base

def voicebank_gen(base_path, output_path):
   

    # 获取路径下所有以.wav结尾的文件
    wav_files = [f for f in os.listdir(base_path) if f.endswith('.wav')]

    # 打开文件用于写入
    with open(output_path, 'w') as f:
        for file in wav_files:
            # 去掉文件扩展名得到文件名主体部分
            file_base = file.replace('.wav', '')
            # 写入文件
            f.write(f"{file_base}|{os.path.join(base_path, file)}\n")

        
if __name__ == "__main__":
    base_path = 'data/voicebank/clean_trainset_wav'
    output_path = 'train_data_fsn_voicebank_master/clean_train.txt'
    voicebank_gen(base_path, output_path)
    base_path = 'data/voicebank/noisy_trainset_wav'
    output_path = 'train_data_fsn_voicebank_master/noisy_train.txt'
    voicebank_gen(base_path, output_path)
    base_path = 'data/voicebank/clean_testset_wav'
    output_path = 'train_data_fsn_voicebank_master/clean_test.txt'
    voicebank_gen(base_path, output_path)

    base_path = 'data/voicebank/noisy_testset_wav'
    output_path = 'train_data_fsn_voicebank_master/noisy_test.txt'
    voicebank_gen(base_path, output_path)