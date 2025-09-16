import numpy as np


#-----------------------------------------------------
#------------------ GENERAL UTILITIES ----------------
#-----------------------------------------------------


def cut_window(lon_min, lon_max, lat_min, lat_max, lon, lat, *argv):
    r'''
    Derives a new version of the longitude, latitude and precipitation
    tensors, by only retaining the values inside the specified lon-lat rectangle
    Args:
        lon_min, lon_max, lat_min, lat_max: integers
        lon, lat, z, pr: np.arrays
    Returns:
        The new tensors with the selected values
    '''

    bool_lon = np.logical_and(lon >= lon_min, lon <= lon_max)
    bool_lat = np.logical_and(lat >= lat_min, lat <= lat_max)
    bool_both = np.logical_and(bool_lon, bool_lat).flatten()
    lon_sel = lon.flatten()[bool_both]
    lat_sel = lat.flatten()[bool_both]
    v = []
    for arg in argv:
        if arg.ndim > 2:
            arg = arg.reshape(arg.shape[0], -1)
            v.append(arg[:, bool_both])
        else:
            arg = arg.flatten()
            v.append(arg[bool_both])
    return lon_sel, lat_sel, *v


#-----------------------------------------------------
#------------------- NODES AND EDGES -----------------
#-----------------------------------------------------


def retain_valid_nodes(pr,mask_land=None):
    r'''
    Selects only the nodes for which precipitation is not
    nan in at least one timestamp. All the other nodes are
    ignored (e.g. the sea values in GRIPHO). If a land mask
    is provided, non-land points are also ignored.
    Args:
        lon (np.array): longitude for each spatial point
        lat (np.array): latitude for each spatial point
        pr (np.array): precipitation for each spatial point
        e (np.array): elevation for each spatial point
        mask_land (np.array, optional): a mask for the land points
    Returns:
        The valid points for each input tensor
    '''

    valid_nodes = ~np.isnan(pr).all(axis=0)
    if mask_land is not None:
        valid_nodes = np.logical_and(valid_nodes, ~np.isnan(mask_land))
    return valid_nodes


def derive_edge_index_within(lon_radius, lat_radius, lon_senders, lat_senders, lon_receivers, lat_receivers, use_edge_attr=True, radius=None):
    r'''
    Derives edge_indexes within two sets of nodes based on specified lon, lat distances
    Args:
        lon_senders (np.array): longitudes of all first nodes in the edges
        lat_senders (np.array): latitudes of all fisrt nodes in the edges
        lon_receivers (np.array): longitudes of all second nodes in the edges
        lat_receivers (np.array): latitudes of all second nodes in the edges
    Return:
        The edge_indexes tensor
    '''

    edge_index = []

    lonlat_senders = np.column_stack((lon_senders, lat_senders))
    lonlat_receivers = np.column_stack((lon_receivers,lat_receivers))

    for ii, xi in enumerate(lonlat_senders):
        
        if radius is not None:
            bool_both = ((lon_receivers - xi[0]) ** 2 + (lat_receivers - xi[1]) ** 2) ** 0.5 < radius
        else:
            bool_lon = np.abs(lon_receivers - xi[0]) < lon_radius
            bool_lat = np.abs(lat_receivers - xi[1]) < lat_radius
            bool_both = np.logical_and(bool_lon, bool_lat)

        jj_list = np.nonzero(bool_both)[0] # to get indices
        xj_list = lonlat_receivers[bool_both]

        for jj, xj in zip(jj_list, xj_list):
            if not np.array_equal(xi, xj):
                edge_index.append(np.array([ii, jj]))
    
    edge_index = np.array(edge_index).T
    print(edge_index.shape)

    return edge_index
