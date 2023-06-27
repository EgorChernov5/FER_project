# FER_project
Исследовательский проект по созданию веб-приложения с внедрением модели машинного обучения по по распознования эмоций человека.

Данный проект позволяет собрать статистические данные об эмоциях человека, которые он испытывает в процессе деятельность. В нашем случае это применимо для введения новой метрики - эмоциональное состояние, в игровой индустрии. 

Использование "FER_project" проекта может иметь множество применений в игровой индустрии и привнести значительные улучшения в игровой бизнес. Вот несколько из них:
- Адаптивный геймплей: Игровые разработчики могут использовать данные об эмоциональном состоянии игрока для создания более адаптивного геймплея. Например, если приложение определяет, что игрок испытывает стресс или фрустрацию, игра может автоматически уменьшить сложность или предложить подсказки, чтобы помочь игроку преодолеть трудности. Это позволит создать более персонализированный и увлекательный опыт для каждого игрока;
- Оценка эмоций игроков: Студии разработки игр и издатели могут использовать данное приложение для сбора данных об эмоциональной реакции игроков на определенные моменты игры, такие как эпические моменты, смешные ситуации или страшные сцены. Эти данные могут быть полезны для оценки эмоциональной привлекательности игры и выявления сильных и слабых сторон. Разработчики смогут проводить более целенаправленное тестирование и внести улучшения, чтобы создать игры, которые вызывают сильные эмоции и удовлетворение у игроков.

**Для создания проекта были пройдены следующие этапы:**
1. Определение целей и ценности проекта;
2. Сбор и обработка данных для модели;
3. Построение базовой модели, визуализация;
4. Обучение модели, обеспечение доступа к ней;
5. Создание веб-приложения;
6. Упаковка MVP в Docker.

**Подробнее про этапы разработки вы можете посмотреть [документацию]().**

**Для запуска проекта необходимо сделать следующее:**
1. Стянуть данный репозиторий;
2. В папке репозитория открыть терминал;
3. Запустить две команды:
    - docker-compose build
    - docker-compose up

После приложение будет доступно по адресу - [http://localhost:8000](http://localhost:8000)
