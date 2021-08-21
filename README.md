# Time Fusion Transformer using PaddlePaddle
### 基于Paddle实现论文: [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf)
### 数据集：[Electricity dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip)

* Convert TXT files to CSV files

  ``` python
  python scripts/convert_data.py
  ```

* Verifying configuration files

  ```python
  python conf/conf.py
  ```

* Verifying dataset

  ```python
  python dataset/ts_dataset.py
  ```

* Training with single GPU

  ```python
  python main.py --exp_name electricity --conf_file_path your_file_path --inference False
  ```

* Inference with best model saved

  ```python
  python main.py --exp_name electricity --conf_file_path your_file_path --inference True
  ```
A Prediction Example
![image](https://github.com/Scallions/tft_paddle/blob/develop/Figure_1.png)
  
