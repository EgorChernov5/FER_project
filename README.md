# FER_project
Исследовательский проект по созданию и внедрению модели машинного обучения по распознования эмоций человека в игровой индустрии.

Применение модели...

Ценность для бизнеса...

**Содержание:**
1. [Prepare data](#prepare-data)
2. [Build model](#build-model)
3. [Prepare data](#deploy-model)

## Prepare data
**Данные для обучения модели:**  
[Emotion Detection](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)  
[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)  
[Rating OpenCV Emotion Images](https://www.kaggle.com/datasets/juniorbueno/rating-opencv-emotion-images)  
[Natural Human Face Images for Emotion Recognition](https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition)  
[Micro_Expressions](https://www.kaggle.com/datasets/kmirfan/micro-expressions)  
[Corrective re-annotation of FER - CK+ - KDEF](https://www.kaggle.com/datasets/sudarshanvaidya/corrective-reannotation-of-fer-ck-kdef)  

## Build model
**Шаги обучения модели:**
- Data collection
- Data cleaning
- Feature engineering
- Model training

**Структура проекта:**
- <kbd>data/</kbd> – для всех версий датасетов
- <kbd>data/prepare/</kbd> – для данных, измененных внутри
- <kbd>data/raw/</kbd> – для данных, полученных из внешнего источника
- <kbd>metrics/</kbd> – для отслеживания показателей производительности моделей
- <kbd>model/</kbd> – для моделей машинного обучения
- <kbd>src/</kbd> – для исходного кода

**Каталог <kbd>src/</kbd> имеет три файла:**
- <kbd>prepare.py</kbd> – код подготовки данных для обучения
- <kbd>train.py</kbd> – код обучения модели
- <kbd>evalueate.py</kbd> – код оценки результатов обучения модели

## Deploy model

