# Fire Detection - Сравнение моделей детекции возгораний

## Датасет

Датасет **D-Fire** не включён в репозиторий.

- Репозиторий датасета: https://github.com/gaia-solutions-on-demand/DFireDataset
- Train / Val / Test сплиты (OneDrive датасета): https://1drv.ms/f/c/c0bd25b6b048b01d/Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA?e=6rk0aN
- Тестовые видео (OneDrive датасета): https://1drv.ms/f/c/c0bd25b6b048b01d/EhT2Jy6L-YlGvZv-gXH2SnYBENQsnUW96LpZtv_6PngjYQ

После скачивания распакуй в корень проекта.

- Датасет D-Fire - оставь название неизменным "..\D-Fire"
- Train / Val / Test сплиты - создай папку "valid" и распакуй в неё содержимое OneDrive
- Тестовые видео - создай папку video и распакуй все видео в неё



## Веса моделей

Обученные веса не включены в репозиторий (хранятся на OneDrive):

| Модель                | OneDrive ссылка                                                                                 |
|-----------------------|-------------------------------------------------------------------------------------------------|
| YOLOv8m               | _https://1drv.ms/f/c/a21603161ce64995/IgD1SZFYIV57T7pPp2QR6BSWAeeDPxIYTk6bq6_saUbuI6c?e=f7x0rn_ |
| YOLOv10s              | _https://1drv.ms/f/c/a21603161ce64995/IgC-4-SH3w68ToNn16PxWtaQAa3CDYJ7g8ql7vPaG-kmK9M?e=7KcQVC_ |
| YOLOv11n              | _https://1drv.ms/f/c/a21603161ce64995/IgCd2i-O1sC0QIfPRsfG6qlBATeuUY65oi19b6dQisixV5A?e=qFxvcf_ |
| RT-DETR-L             | _https://1drv.ms/f/c/a21603161ce64995/IgBhpuU5ElzcSZJ2bg-L4-V_AXPcoqRZxT80F-Ek_-ix-bQ?e=EHiHRj_ |
| Faster R-CNN ResNet50 | _https://1drv.ms/f/c/a21603161ce64995/IgBk9h0q6vZfQLctlSNimkplAaRN6mZlaD72BkjaAhgTFuQ?e=OkajMX_ |
| ONNX и Torchscript    | _https://1drv.ms/f/c/a21603161ce64995/IgAbFcjJlgGXRozTgYCMLLEmAQUQKC2Keit8hPspaZ6vo_c?e=W1qdf5_ |
| Best model (service)  | _https://1drv.ms/u/c/a21603161ce64995/IQDDqDAg1-l0SZgRSxZLcSrGASbkqpHLb3vW3XmGPUJOy3E_          |

После скачивания помести веса по путям:
- `outputs/runs/<model_name>/weights/best.pt`
- `models/best_model/weights.pt` - в OneDrive вес имеет имя `best.pt`, необходимо переименовать в `weights.pt`, как указано в пути
- `models/exported`- содержит экспортированные веса ONNX и TorchScript, экспорт реализован в `export_yolo_onnx.py`

Базовые веса YOLO/RT-DETR (pretrained) загружаются автоматически
при первом запуске через `ultralytics`.