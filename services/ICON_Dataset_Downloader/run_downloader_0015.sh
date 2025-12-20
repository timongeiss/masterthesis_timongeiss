#!/bin/bash

LOGFILE="/volume2/TimonDWD/dwd_downloader_timon/scheduler.log"
WORKDIR="/volume2/TimonDWD/dwd_downloader_timon"

{
    echo "🕕 Starte Task: $(date)"

    cd "$WORKDIR" || { echo "❌ Verzeichnis nicht gefunden: $WORKDIR"; exit 1; }

    # Gestern im UTC-Format
    #DATE=$(date -u -d "yesterday" +%Y-%m-%d)
    DATE=$(date -d "yesterday" +%Y-%m-%d)


    echo "🕕 Erzeuge .env für 00:15 Uhr Download (UTC-Läufe vom Vortag: $DATE)"
    cat <<EOF > .env
TIMESTAMP1=${DATE}T06:00:00
TIMESTAMP2=${DATE}T09:00:00
TIMESTAMP3=${DATE}T12:00:00
TIMESTAMP4=${DATE}T15:00:00
EOF

    echo "✅ .env erzeugt:"
    cat .env

    echo "🛠 docker-compose down"
    docker-compose down || echo "⚠️ compose down fehlgeschlagen (evtl. kein aktiver Container)"

    echo "🚀 docker-compose up -d"
    docker-compose up -d

    echo "✅ Task beendet: $(date)"
} >> "$LOGFILE" 2>&1
