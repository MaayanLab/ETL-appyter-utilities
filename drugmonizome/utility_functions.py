# Adapted from code created by Moshe Silverstein

import datetime
import os
import zipfile

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.sparse as sp

from tqdm import tqdm



def merge(df, axis):
    '''
    Merges duplicate rows or columns, depending on the axis specified. The
    final values of the merged rows or columns is determined by the method.
    '''
    if axis == 'column':
        return df.groupby(df.columns, axis=1).mean()
    elif axis == 'row':
        return df.groupby(level=0, axis=0).mean()

def map_symbols(df, symbol_lookup, remove_duplicates=False):
    '''
    Replaces the zeroth index column of the df, which are gene names, with
    corresponding approved gene symbols according to the given symbol_lookup 
    dictionary. If any gene names are not in the mapping, they are discarded 
    from the DataFrame.
    '''
    tqdm.pandas()

    df.iloc[:, 0] = df.iloc[:, 0].progress_map(
        lambda x: symbol_lookup.get(x, np.nan))

    df = df.dropna(subset=[df.columns[0]])
    if remove_duplicates:
        df = df.drop_duplicates()
    return df

def binary_matrix(df):
    '''
    Creates an adjacency matrix from df, which is a drug-attribute edge
    list.
    '''
    matrix = pd.crosstab(df.index, df.iloc[:, 0]) > 0
    matrix.index.name = df.index.name
    matrix.columns.name = df.columns[0]
    return matrix

def file_name(path, name, ext):
    '''
    Returns the file name by taking the path and name, adding the year and month
    and then the extension. The final string returned is thus
        '<path>/<name>_<year>_<month>.ext'
    '''
    date = str(datetime.date.today())[0:7].replace('-', '_')
    filename = ''.join([name, '_', date, '.', ext])
    return os.path.join(path, filename)


def save_setlib(df, lib, path, name):
    '''
    If lib = 'drug', this creates a GMT file which lists all attributes and the
    drugs that are associated with that attribute.

    If lib = 'attribute', this creates a GMT file which lists all drugs and the
    attributes that are associated with that drug.

    The year and month are added at the end of the name. The path the file is
    saved to is thus
        path + name + '_<year>_<month>.gmt'

    All set sizes must be greater than five to be exported!
    '''
    filenameGMT = file_name(path, name, 'gmt')

    if not (lib == 'drug' or lib == 'attribute'):
        return
    if lib == 'attribute':
        df = df.T

    with open(filenameGMT, 'w') as f:
        arr = df.reset_index(drop=True).to_numpy(dtype=int)
        attributes = df.columns

        w, h = arr.shape
        for i in tqdm(range(h)):
            if len(set(df.index[arr[:, i] == 1])) >= 5:
                print(attributes[i], *df.index[arr[:, i] == 1],
                    sep='\t', end='\n', file=f)
            else:
                pass


def similarity_matrix(df, metric, dtype=None, sparse=False):
    '''
    Creates a similarity matrix between the rows of the df based on
    the metric specified. The resulting matrix has both rows and columns labeled
    by the index of df.
    '''
    if sparse and metric == 'jaccard':
        # from na-o-ys on Github
        sparse = sp.csr_matrix(df.to_numpy(dtype=bool).astype(int))
        cols_sum = sparse.getnnz(axis=1)
        ab = sparse * sparse.T
        denom = np.repeat(cols_sum, ab.getnnz(axis=1)) + \
            cols_sum[ab.indices] - ab.data
        ab.data = ab.data / denom
        similarity_matrix = ab.todense()
        np.fill_diagonal(similarity_matrix, 1)

    else:
        similarity_matrix = dist.pdist(df.to_numpy(dtype=dtype), metric)
        similarity_matrix = dist.squareform(similarity_matrix)
        similarity_matrix = 1 - similarity_matrix

    similarity_df = pd.DataFrame(
        data=similarity_matrix, index=df.index, columns=df.index)
    similarity_df.index.name = None
    similarity_df.columns.name = None
    return similarity_df

def save_data(df, path, name, compression=None, ext='tsv',
              symmetric=False, dtype=None, **kwargs):
    '''
    Save df according to the compression method given. 
    compression can take these values:
        None or 'gmt' - defaults to pandas to_csv() function.
        'gzip' - uses the gzip compression method of the pandas to_csv() function
        'npz' - converts the DataFrame to a numpy array, and saves the array.
                The array is stored as 'axes[0]_axes[1]'. If symmetric is true,
                it is stored as 'axes[0]_axes[1]_symmetric' instead.
    ext is only used if compression is None or 'gzip'. The extension of the file
    will be .ext, or .ext.gz if 'gzip' is specified.
    axes must only be specified if compression is 'npz'. It is a string tuple
    that describes the index and columns df, i.e. (x, y) where x, y = 
    'drug' or 'attribute'.
    symmetric is only used if compression is 'npz', and indicates if df
    is symmetric and can be stored as such. 
    dtype is only used if compression is 'npz', and indicates a dtype that the
    array can be cast to before storing.

    The year and month are added at the end of the name. The path the file is 
    saved to is thus
        path + name + '_<year>_<month>.ext'
    where ext is .ext, .ext.gz, or .npz depending on the compression method.
    '''

    if compression is None:
        name = file_name(path, name, ext)
        df.to_csv(name, sep='\t', **kwargs)
    elif compression == 'gzip':
        name = file_name(path, name, ext + '.gz')
        df.to_csv(name, sep='\t', compression='gzip', **kwargs)
    elif compression == 'npz':
        name = file_name(path, name, 'npz')

        data = df.to_numpy(dtype=dtype)
        index = np.array(df.index)
        columns = np.array(df.columns)

        if symmetric:
            data = np.triu(data)
            np.savez_compressed(name, symmetric=data, index=index)
        else:
            np.savez_compressed(name, nonsymmetric=data,
                                index=index, columns=columns)


def load_data(filename):
    '''
    Loads a pandas DataFrame stored in a .npz data numpy array format.
    '''
    with np.load(filename, allow_pickle=True) as data_load:
        arrays = data_load.files
        if arrays[0] == 'symmetric':
            data = data_load['symmetric']
            index = data_load['index']
            data = data + data.T - np.diag(data.diagonal())
            df = pd.DataFrame(data=data, index=index, columns=index)
            return df
        elif arrays[0] == 'nonsymmetric':
            data = data_load['nonsymmetric']
            index = data_load['index']
            columns = data_load['columns']
            df = pd.DataFrame(data=data, index=index, columns=columns)
            return df


def archive(path):
    with zipfile.ZipFile('output_archive.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(path):
            for f in files:
                zipf.write(os.path.join(root, f))
