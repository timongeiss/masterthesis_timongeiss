FROM continuumio/miniconda3:24.3.0-0

# Installiere benötigte Pakete aus conda-forge
RUN conda install -c conda-forge \
    python=3.10 \
    cfgrib \
    xarray \
    pandas \
    numpy \
    eccodes \
    tqdm \
    pip && \
    conda clean --all -f -y

# Optional: Python-Pakete aus requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

#COPY config.txt /app/config.txt

# Kopiere restlichen Code und setze Arbeitsverzeichnis
COPY . /app
WORKDIR /app

# Startkommando
CMD ["python", "ICON_Dataset_Extractor.py"]
