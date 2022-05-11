#FROM rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04-py3.8

FROM rapidsai/rapidsai-core:22.04-cuda11.5-runtime-ubuntu20.04-py3.9

RUN source /opt/conda/bin/activate && conda activate rapids && conda install --name rapids pylint -y \
    && cd /tmp/ && wget https://interactivebrokers.github.io/downloads/twsapi_macunix.981.01.zip \
    && unzip twsapi_macunix.981.01.zip \
    && cd /tmp/IBJts/source/pythonclient \
    && python3 setup.py build install \
    && pip install influxdb_client \
    && pip install backtrader \
    && conda install sqlalchemy \
    && conda install kafka-python \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt update \
    && apt-get install redis-server -y \
    && ACCEPT_EULA=Y apt-get install msodbcsql18 -y\
    && printf "[MongoSQL]\nDRIVER=ODBC Driver 18 for SQL Server\nSERVER=tcp:192.168.2.40,1433" >> /etc/odbc.ini \
    && conda install pyodbc \
