import time
import numpy as np
import asyncio


async def async_io_test(t):
    await asyncio.sleep(t)


def io_test(t):
    time.sleep(t)


def cpu_test(times):
    count = 0
    for i in range(times):
        count += 1


def get_cur_millisecond():
    return int(time.time() * 1000)


def generate_urls(url, ports):
    url_list = list()
    port_list = ports.split(',')
    base_port = port_list[0]
    for port in port_list:
        url_list.append(url.replace(base_port, port))
    return port_list, url_list


def calc_time_p99(time_list):
    time_num = len(time_list)
    time_list.sort()

    # print(time_list)

    min = time_list[0]
    max = time_list[-1]

    p25 = time_list[int(time_num * 0.25)]
    p50 = time_list[int(time_num * 0.50)]
    p75 = time_list[int(time_num * 0.75)]
    p99 = time_list[int(time_num * 0.99)]
    p999 = time_list[int(time_num * 0.999)]

    avg = np.mean(time_list)

    print("[总数] %d, [min] %d ms, [max] %d ms, [average] %f ms," % (time_num, min, max, avg.item()))
    print("[p25] %d ms, [p50] %d ms, [p75] %d ms, [p99] %d ms, [p999] %d ms" % (p25, p50, p75, p99, p999))
    print("[p99-average] %f, [p99/average] %f" % (p99-avg.item(), float(p99)/avg.item()))


def calc_success_rate(reslut_list):
    num = len(reslut_list)
    print("[总数] %d [成功率] %f [失败数] %d" % (num, float(reslut_list.count(200))/float(num), num-reslut_list.count(200)))