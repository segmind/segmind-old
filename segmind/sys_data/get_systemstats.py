import psutil
from psutil._common import bytes2human


def fan_stats():
    data = {}
    if not hasattr(psutil, 'sensors_fans'):
        return data
    fans = psutil.sensors_fans()
    if fans:
        for name, entries in fans.items():
            for entry in entries:
                data[entry.label or name] = entry.current
    return data


def bits_to_values(nt):

    data = {}
    for name in nt._fields:
        value = getattr(nt, name)
        if name != 'percent':
            value = bytes2human(value)
        data[name.capitalize()] = value
    return data


def system_memory_stats():
    return bits_to_values(psutil.virtual_memory())


def system_temp_stats():
    data = {}
    temps = psutil.sensors_temperatures()
    avg_temp = 0
    N = 1
    for name, entries in temps.items():
        if name.startswith('coretemp'):
            for entry in entries:
                avg_temp += entry.current
                N += 1
    data['avg_CPU_temp'] = avg_temp/N
    return data


def battery_stats():

    data = {}
    if not hasattr(psutil, 'sensors_battery'):
        return data
    batt = psutil.sensors_battery()
    if batt is not None:
        data['battery_percentage'] = round(batt.percent, 2)
        data['battery_seconds_left'] = batt.secsleft
    else:
        data['battery_percentage'] = 100

    return data


def cpu_usage_stats():
    data = {}
    # usage_list = psutil.cpu_percent(interval=1.0, percpu=True)

    # for index, value in enumerate(usage_list):
    #     data[f'CPU{index}_usage'] = value
    data['avg_CPU_usage'] = psutil.cpu_percent(interval=1.0)

    return data


def system_metrics():
    data = system_temp_stats()

    memory_data = system_memory_stats()
    data['RAM_usage'] = memory_data['Percent']
    data.update(battery_stats())
    data.update(cpu_usage_stats())
    data.update(fan_stats())

    return data
