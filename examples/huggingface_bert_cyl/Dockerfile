FROM swr.cn-central-221.ovaijisuan.com/wair/pytriton:base_py39_torch2.3

COPY huggingface_bert_cyl /home/server/huggingface_bert_cyl

WORKDIR /home/server/huggingface_bert_cyl

CMD ["python", "server_embed.py"]

