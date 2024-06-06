FROM swr.cn-central-221.ovaijisuan.com/wair/tritonserver:24.04-trtllm

USER root

# COPY ./bce_all_gpu/requirements.txt /tmp/requirements.txt

# RUN pip3 install pip -U -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
#     && pip3 install -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

WORKDIR /home/server

COPY . .

CMD ["bash", "scripts/run_server.sh"]