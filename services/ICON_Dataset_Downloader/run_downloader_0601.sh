#!/bin/bash

LOGFILE="/volume2/TimonDWD/dwd_downloader_timon/scheduler.log"
WORKDIR="/volume2/TimonDWD/dwd_downloader_timon"

{
    echo "🕕 Starte Task: $(date)"

    cd "$WORKDIR" || { echo "❌ Verzeichnis nicht gefunden: $WORKDIR"; exit 1; }

    TODAY=$(date -u +%Y-%m-%d)
    YESTERDAY=$(date -u -d "yesterday" +%Y-%m-%d)

    echo "🕕 Erzeuge .env für 06:00 Uhr Download"
    cat <<EOF > .env
TIMESTAMP1=${YESTERDAY}T18:00:00
TIMESTAMP2=${YESTERDAY}T21:00:00
TIMESTAMP3=${TODAY}T00:00:00
TIMESTAMP4=${TODAY}T03:00:00
EOF

    echo "✅ .env erzeugt:"
    cat .env

    echo "🛠 docker-compose down"
    docker-compose down || echo "⚠️ compose down fehlgeschlagen (evtl. kein aktiver Container)"

    echo "🚀 docker-compose up -d"
    docker-compose up -d

    echo "✅ Task beendet: $(date)"
} >> "$LOGFILE" 2>&1
