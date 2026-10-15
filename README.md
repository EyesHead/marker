# Marker: PDF → Markdown с HTML-таблицами

Форк [EyesHead/marker](https://github.com/EyesHead/marker) для PDF, включая русскоязычные СП и ГОСТ. Обычный **`marker_single input.pdf --use_llm`** использует исходную разметку Marker и **Ollama Cloud**. Модель по умолчанию — **`kimi-k3`**: она принимает изображения и имеет контекст 1 млн токенов. Это выбор с приоритетом качества; преимущество на конкретном PDF нужно проверять на образцах. [Карточка модели](https://ollama.com/library/kimi-k3:cloud).

В карточке Ollama указана цена $3 за 1 млн входных и $15 за 1 млн выходных токенов. Это более дорогой вариант; для пробы используйте `--max-requests 12`. Лимит учитывает повторы, но ограничивает число запросов, а не сумму в долларах.

## Что получается

- Markdown с семантическими HTML-таблицами: строки, заголовки, `rowspan`, `colspan`.
- Продолжения одной таблицы объединяются до распознавания. Все её части передаются в одном задании модели вместе с координатами ячеек, отдельных слов и сведениями о базовых линиях индексов; при необходимости в тот же запрос добавляются крупные кропы целых выражений.
- Формулы в ячейках — **MathML с исходным LaTeX в `<annotation encoding="application/x-tex">`**. Вне таблиц Markdown содержит LaTeX. MathML строится локально из той же строки LaTeX.
- Схемы и фотографии сохраняются в PNG из исходного PDF и вставляются в соответствующие ячейки. Модель определяет размещение; пиксели берутся из PDF.

Локально выполняются извлечение текста, геометрии и лёгкое определение разметки. **Локальная большая языковая модель и сервер Ollama не требуются.** Сканы и повреждённые области направляются в облако. Растровые и векторные объекты вне найденных блоков дополнительно проверяются на покрытие. Для сканов без надёжной разметки сначала определяются целые объекты, затем объединяются продолжения таблиц.

Каждая таблица распознаётся отдельно. HTML проверяется на целостность сетки, допустимые объединения и сохранность доступного исходного текста. Видимые индексы проверяются по доступной геометрии. Ошибка обязательной таблицы или формулы останавливает публикацию результата: её PNG не считается успешно распознанной таблицей или формулой. Эти проверки не доказывают безошибочность визуального распознавания.

## Папки

```text
PolytechBrain/
  ScriptOllama.py
  ScriptOllama.sh            # macOS / Linux
  ScriptMacOS-Ollama.sh     # прежнее имя для macOS
  ScriptOllama.cmd          # Windows
  ScriptOllama.ps1          # PowerShell, необязательно
  PDF/
    document.pdf
  Output/
    document/
      document.md
      document_meta.json
      ...PNG рисунков...
```

Папки `PDF` и `Output` находятся рядом с `ScriptOllama.py`; расположение хранилища не меняется. Обрабатываются `.pdf`/`.PDF` непосредственно в `PDF`. Пути с пробелами и кириллицей поддерживаются. Исходные PDF сохраняются.

## Быстрый запуск

Нужны Python 3.10+, Git, интернет и [API-ключ Ollama](https://ollama.com/settings/keys). Аккаунт должен предоставлять доступ к выбранной модели. При отсутствии доступа скрипт сообщает об ошибке и сохраняет прогресс.

### macOS

Выполните в папке хранилища:

```bash
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.py -o ScriptOllama.py
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.sh -o ScriptOllama.sh
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptMacOS-Ollama.sh -o ScriptMacOS-Ollama.sh
chmod +x ScriptOllama.sh ScriptMacOS-Ollama.sh
./ScriptOllama.sh
```

Ключ вводится в отдельном системном окне со скрытым вводом. При недоступности окна используется скрытый ввод в терминале. Прежняя команда `./ScriptMacOS-Ollama.sh` также работает.

### Ubuntu 25

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev git curl build-essential
# Далее в папке хранилища:
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.py -o ScriptOllama.py
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.sh -o ScriptOllama.sh
chmod +x ScriptOllama.sh
./ScriptOllama.sh
```

### Windows 10/11

Установите Python 3.10+ с Python Launcher (`py`) или Python в `PATH`, а также Git. В PowerShell, в папке хранилища:

```powershell
Invoke-WebRequest 'https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.py' -OutFile ScriptOllama.py
Invoke-WebRequest 'https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.cmd' -OutFile ScriptOllama.cmd
.\ScriptOllama.cmd
```

На Linux и Windows ключ вводится скрыто в терминале; без терминала используется окно Tk, если оно доступно. Отмена ввода прекращает запуск. Значение `OLLAMA_API_KEY` в окружении заменяет диалог. Ключ не передаётся в аргументах и не сохраняется в настройках или результатах.

## Сценарии запуска

```bash
# Обычный запуск: Kimi K3 в облаке
./ScriptOllama.sh

# Обновить установленный Marker и пересобрать готовые заметки
./ScriptOllama.sh --update-marker --force

# PDF с хорошим текстовым слоем: сохранить обычный текст, распознавать математику и таблицы
./ScriptOllama.sh --disable-ocr --force

# Заново распознавать текст сканов/повреждённых PDF
./ScriptOllama.sh --force-ocr --force

# Явно выбрать другую модель Ollama Cloud
./ScriptOllama.sh --model gemma4:31b --force

# Ограниченный пробный прогон: не более 12 запросов на PDF, включая повторы
./ScriptOllama.sh --max-requests 12 --force
```

Windows: те же параметры после `.\ScriptOllama.cmd`. Параметры `--force-ocr` и `--disable-ocr` несовместимы. При ограничении запросов и незавершённом распознавании результат не публикуется; следующий запуск продолжает с сохранённых успешных объектов.

## Запуск установленного Marker напрямую

После установки этого форка и задания `OLLAMA_API_KEY` в окружении:

```bash
marker_single input.pdf --use_llm --output_format markdown --output_dir Output

marker_single input.pdf --use_llm \
  --ollama_model kimi-k3 \
  --ollama_base_url https://ollama.com/v1 \
  --cloud_work_dir Output/.marker-work \
  --cloud_max_requests 12
```

Тот же путь по умолчанию доступен через `PdfConverter(artifact_dict=models, config={"use_llm": True})`. Явный `--llm_service` сохраняется как переопределение провайдера. Явный список `--processors` задаёт собственный конвейер и заменяет стандартный набор обработчиков.

## Установка и повторный запуск

Первый запуск клонирует репозиторий в отдельный кэш, создаёт `venv` и устанавливает Marker. Повторные запуски используют установленную ревизию без повторного клонирования. `--update-marker` обновляет её явно; прежняя установка без нового native-обработчика обновляется однократно. Локально изменённый кэш автоматически не перезаписывается.

По умолчанию используется ветка `codex/my-change`. После её слияния можно явно выбрать `--ref master --update-marker`. Обновление Marker не заменяет файлы запускателя: их нужно скачать повторно командами выше.

Кэш: macOS — `~/Library/Caches/eyeshead-marker` (существующий `~/.cache/eyeshead-marker` сохраняется); Linux — `$XDG_CACHE_HOME/eyeshead-marker` или `~/.cache/eyeshead-marker`; Windows — `%LOCALAPPDATA%\eyeshead-marker`. Путь переопределяется через `--cache-dir` / `MARKER_CACHE`.

Успешные задания и исходные PNG для проверки находятся в `Output/.marker-work/<идентификатор>/`. Кэш зависит от PDF, модели, адреса сервиса и параметров. Окончательная заметка содержит PNG рисунков; исходные кропы распознанных формул остаются в рабочем кэше. `--force` пересобирает заметку, используя подходящие успешные задания. Прежняя заметка сохраняется до успешного завершения нового результата.

## Основные параметры

| Параметр запускателя | Переменная окружения | По умолчанию |
|---|---|---|
| `--model` | `OLLAMA_MODEL` | `kimi-k3` |
| `--base-url` | `OLLAMA_BASE_URL` | `https://ollama.com/v1` |
| `--context-tokens` | `CLOUD_CONTEXT_TOKENS` | `245760` |
| `--max-output-tokens` | `CLOUD_MAX_OUTPUT_TOKENS` | `16384` |
| `--max-images` | `CLOUD_MAX_IMAGES` | `16` |
| `--crop-dpi` | `CLOUD_CROP_DPI` | `600` |
| `--max-attempts` | `CLOUD_MAX_ATTEMPTS` | `3` |
| `--max-concurrency` | `CLOUD_MAX_CONCURRENCY` | `2`, диапазон 1–4 |
| `--max-requests` | `CLOUD_MAX_REQUESTS` | `0`, без общего лимита |

Контекст — консервативный клиентский бюджет, а не настройка контекста сервера. Изображения для запросов уменьшаются отдельно от сохраняемых исходных ресурсов. Дополнительные кропы формул добавляются только при сохранении разрешения основных изображений. Если целая таблица превышает бюджет, распознавание остаётся незавершённым с явной причиной. Увеличьте соответствующий лимит и повторите запуск.

## Проверки

```bash
python -m unittest discover -s tests/cloud -v
python -m unittest discover -s tests/html_tables -v
```

Офлайн-тесты используют синтетические PDF и подменённые ответы, не вызывая Ollama и не скачивая модели. Они проверяют конвейер, сохранность исходных ресурсов и обработку ошибок. Точность выбранной модели требует отдельной проверки реальных ответов на конкретных документах. CI настроен для macOS, Ubuntu и Windows; запуск тестов на одной ОС не подтверждает остальные.
