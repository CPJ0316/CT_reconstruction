比較CT影像的重建在多種變因下的成果差異。
- 各檔案簡易說明(比較的結果出自CT_reconstruction.m、CT_reconstruction_self.py)
  - CT_reconstruction.m: 使用Matlab的API進行影像重建
  - CT_reconstruction.py: 使用Python的API進行影像重建
  - CT_reconstruction_self.py: 使用Python並自行撰寫程式進行影像重建
- 重建間隔角度的差異成果比較
  - 角度間隔越小，重建所使用到的影像用多，所重建出的影像會越接近真實影像，且由於 pixel的疊加值增加，因此重建後的影像亮度較高。
<img width="1207" height="664" alt="image" src="https://github.com/user-attachments/assets/e5958226-9d78-4b1b-94e5-7c057e55642f" />

- 有無filter、filter在 frequency domain與 spatial domain的比較
  - 有使用 filter的影像會比沒有使用 filter的影像更為清楚，高頻更加明顯，畫面的整體輪廓也更清晰 。
<img width="1199" height="601" alt="image" src="https://github.com/user-attachments/assets/6c732fd2-fb38-40b3-8f0b-1d03b3a27c9f" />

-使用matlab的function、python實作的結果比較
  - 重建品質為 Matlab 的更勝一籌，且以執行時間來看， 相比使用Matlab library的 function，自行撰寫的code顯然需要耗費好幾倍的時間。 推測除了語言不同的原因之外也可能是因
為 library的 function 有進行一定程度的優化 。
<img width="1295" height="407" alt="image" src="https://github.com/user-attachments/assets/4bf518c1-e9d2-4d9f-b35b-fb977aeb365f" />
