# Competition-AICUP2023-MultimodalClassificationOfVoicePathology

[競賽網站](https://tbrain.trendmicro.com.tw/Competitions/Details/27)

---

- 安裝環境：
    1. 使用 [Anaconda](https://www.anaconda.com/download) 安裝 `Python 3.9.15`
    並安裝 `Jupyter Lab` 或 `Jupyter Notebook`
    2. 安裝 `PyTorch 1.13.1` 以及 `CUDA 11.7` 可從[官方網站](https://pytorch.org/get-started/previous-versions/)找到指令，或直接輸入指令：  
    `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
    3. 使用指令 `pip3 install -r requirements.txt` 來安裝所需套件

+ 資料前處理：
    1. 先將原始音檔 (`.wav`) 分別放進資料夾
        - 訓練集：`dataset\training\training_voice_data`
        - public測試集：`dataset\test\test_data_public`
        - private測試集：`dataset\test\test_data_private`
    2. 分別使用以下程式來生成各自的 Numpy 壓縮檔案 (`.npz`)，該檔案會直接包含 `.wav` 的訊號與 `.csv` 內對應的資訊
        - 訓練集：`processing\training-make_npz.ipynb`
        - public測試集：`processing\test_public-make_npz.ipynb`
        - private測試集：`processing\test_private-make_npz.ipynb`
    3. 分別使用以下程式對剛剛的 `.npz` 裡的訊號進行快速傅立葉轉換 (FFT)，並另存檔案
        - 訓練集：`processing\training-fft.ipynb`
        - public測試集：`processing\test_public-fft.ipynb`
        - private測試集：`processing\test_private-fft.ipynb`
    4. (選用) 分別使用以下程式來生成記錄每個 `.npz` 檔名的 `.json` 檔案，以方便後續抓取檔案，  
    其中`訓練集`會隨機抽選訓練與測試集，如欲重新抽選檔案可修改程式內的`種子碼`來重新抽選
        - 訓練集：`processing\training-make_data_list.ipynb`
        - public測試集：`processing\test_public-make_data_list.ipynb`
        - private測試集：`processing\test_private-make_data_list.ipynb`
    
- 訓練方式：
    1. 開啟 `experiments\traininig.ipynb`
    2. 從第`2`個儲存格開始執行程式，至第`7`格為止
    3. 第`8`格可設定模型的名稱，此功能是用於`重新訓練`的，如果要把已經訓練好的模型載入回來，就不需要執行這格
    4. 第`9`格用於載入`已訓練的模型`，需手動修改模型名稱，預設為我們訓練好的模型 `m202_MultiOutput_20230518_114418`
    5. 執行過第`8`或第`9`格以後，一直往下執行至第`15`格為止
    6. 第`16`格可用來`重新訓練`模型，第`17`格為`繼續訓練`模型
    7. 第`18`格用來定義繪製 `loss` 與 `UAR` 曲線的函式
    8. 如果為重新訓練模型，執行第`19`格可看到曲線；如果是繼續訓練的模型，則要執行第`20`格
    9. 從第`21`格一直執行至第`25`格為止，可從第`25`格的輸出看到每個保存的模型在訓練與測試集中，三個輸出對應的 `acc` 與 `UAR`
    10. 第`26`格用於讀回剛剛測試的結果，節省以後還要重跑結果的時間
    11. 第`27~29`格用來觀察各個模型在訓練與測試集中，三個輸出對應的`混淆矩陣`以及其他數據，  
    預設要觀察的模型為各個輸出在訓練過程中獲得`最高UAR`的模型 `best_uar-0`、`best_uar-1` 與 `best_uar-2`
    12. 第`30`格用來查看`tensorboard`所記錄的資訊

+ 測試方式：
    1. 開啟 `experiments\test.ipynb`
    2. 從第`2`個儲存格開始執行程式，至第`6`格為止
    3. 在第`7`格設定要測試的模型名稱，預設為我們訓練好的模型 `m202_MultiOutput_20230518_114418`
    4. 從第`8`格一直執行到第`11`格為止
    5. 第`12`格可設定要用於`測試的模型`以及模型的`輸出位置`，我們設定為`best_uar-0`以及第`0`個輸出位置