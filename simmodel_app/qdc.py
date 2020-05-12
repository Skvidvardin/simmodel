import numpy as np


def next_c(t, k_knot_locations, k_servers_activity_indicators):
    l = np.min(np.argwhere(k_knot_locations > t))  # Find l such that x_l < t ≤ x_(l+1)
    if k_servers_activity_indicators[l] == 0:
        return k_knot_locations[l]
    else:
        return t


def qdc_main(arrival_serving_time, knot_locations, servers_activity_indicators):
    """

    :param arrival_serving_time:
    :param knot_locations:
    :param servers_activity_indicators:
    :return:
    """
    arrival_serving_time.insert(0, np.arange(len(arrival_serving_time)), )
    arrival_serving_time = arrival_serving_time[arrival_serving_time[:, 1].argsort()]  # Sort (a,s) in terms of a(ascending)

    for k in knot_locations:  # x_k,L_(k+1) ← oo
        k.append(np.inf)

    for k in servers_activity_indicators:  # y_k,L_(k+2) ← 1
        k.append(np.inf)

    K = len(servers_activity_indicators)  # K ← length(x)

    b = np.zeros(K)
    c = np.array(K)

    p = np.array(len(arrival_serving_time))
    d = np.array(len(arrival_serving_time))

    for i in range(len(arrival_serving_time)):
        for k in range(K):
            c[k] = next_c(max(arrival_serving_time[i, 1], b[k]), knot_locations[k], servers_activity_indicators[k])

        p[i] = np.argmin(b)
        b[p[i]] = c[p[i]] + arrival_serving_time[i, 2]
        d[i] = b[p[i]]

    arrival_serving_time = arrival_serving_time[arrival_serving_time[:, 0].argsort()]

    return -1