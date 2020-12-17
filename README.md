# deeppacket
This is an unofficial implemetation for the paper "Deep Packet: A Novel Approach For Encrypted Traffic
Classification Using Deep Learning" https://arxiv.org/pdf/1709.02656.pdf.

## Enviroment：
```
    OS:	Ubuntu 14.04.5 LTS
    CPU:	Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz (80顆CPU)
    Mem:	503G
    GPU:	Tesla P100-SXM2-16GB

    Python 3.6.3 |Anaconda, Inc.| (default, Oct 13 2017, 12:02:49) [GCC 7.2.0] on linux
    scapy==2.4.0
    numpy==1.13.3
    scikit-learn==0.19.1
    Keras==2.1.3
    tensorflow==1.3.0
```
## Preprocessing:
請助教至 https://drive.google.com/file/d/1vu79_SJoKbqMvdoK3Y7wXti1wUp5qzgw/view
    
下載完整的 pcap 資料夾(內含150個 .pcap 或 .pcapng 檔)，打開 prepro.py，在 line 101 找到 todo_list = gen_todo_list(‘../pcaps’) ，將’../pcaps’ 改成助教下載的 pcap 資料夾路徑之後，執行 
``` shell
python prepro.py
```

運行結束後，應該會產生300個 .pickle 檔，將 .pcap 檔案前處理成 numpy array。後綴 ‘_class’ 是表示 characterization 的 labels。
    
由於 pcap 太大，有些環境沒辦法成功 preprocessing，如果助教有問題的話可以再聯絡我們，我們再上傳我們 preprocess 好的 .pickle 檔案讓助教執行。


## Training/Testing:
```shell
python main.py [-n <model name>] [-t <model type>] [-tt <task type>] [-m <mode>] [-bs <batch size>]
```
參數(詳見 utils.py: get_args() )：





### Training
分別運行以下四條就可以 train 出四個 model，每個 training 結果和過程都會保存在 models/ 底下
```shell
python main.py -n cnn -t cnn -tt app -m train
python main.py -n sae -t sae -tt app -m train
python main.py -n cnn_c -t cnn -tt class -m train
python main.py -n sae_c -t cnn -tt class -m train
```
### Testing
分別運行以下四條就可以 test
```shell
python main.py -n cnn -t cnn -tt app -m test
python main.py -n sae -t sae -tt app -m test
python main.py -n cnn_c -t cnn -tt class -m test
python main.py -n sae_c -t cnn -tt class -m test
```

我們有將 model 保留下來，如果助教想直接 load 我們的 model，可以再聯絡我們，我們再將 model 上傳給助教下載。
