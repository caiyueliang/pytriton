FROM swr.cn-central-221.ovaijisuan.com/wair/pytriton:base_py39_torch2.3

COPY huggingface_bert_cyl /home/server/huggingface_bert_cyl

WORKDIR /home/server/huggingface_bert_cyl

ENV LD_LIBRARY_PATH=/usr/local/python3.9.10/lib:$LD_LIBRARY_PATH

CMD ["python", "server_embed.py"]

