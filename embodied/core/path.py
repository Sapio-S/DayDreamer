import contextlib
import glob
import os
import re
import shutil


class Path:

  filesystems = []

  def __new__(cls, path):
    path = str(path)
    for impl, pred in cls.filesystems:
      if pred(path):
        obj = super().__new__(impl)
        obj.__init__(path)
        return obj
    raise NotImplementedError(f'No filesystem supports: {path}')

  def __getnewargs__(self):
    return (self._path,)

  def __init__(self, path):
    assert isinstance(path, str)
    path = re.sub(r'^\./*', '', path)  # Remove leading dot or dot slashes.
    path = re.sub(r'(?<=[^/])/$', '', path)  # Remove single trailing slash.
    path = path or '.'  # Empty path is represented by a dot.
    self._path = path

  def __truediv__(self, part):
    sep = '' if self._path.endswith('/') else '/'
    return type(self)(f'{self._path}{sep}{str(part)}')

  def __repr__(self):
    return f'Path({str(self)})'

  def __fspath__(self):
    return str(self)

  def __eq__(self, other):
    return self._path == other._path

  def __lt__(self, other):
    return self._path < other._path

  def __str__(self):
    return self._path

  @property
  def parent(self):
    if '/' not in self._path:
      return type(self)('.')
    parent = self._path.rsplit('/', 1)[0]
    parent = parent or ('/' if self._path.startswith('/') else '.')
    return type(self)(parent)

  @property
  def name(self):
    if '/' not in self._path:
      return self._path
    return self._path.rsplit('/', 1)[1]

  @property
  def stem(self):
    return self.name.split('.', 1)[0] if '.' in self.name else self.name

  @property
  def suffix(self):
    return ('.' + self.name.split('.', 1)[1]) if '.' in self.name else ''

  def read(self, mode='r'):
    assert mode in 'r rb'.split(), mode
    with self.open(mode) as f:
      return f.read()

  def write(self, content, mode='w'):
    assert mode in 'w a wb ab'.split(), mode
    with self.open(mode) as f:
      f.write(content)

  @contextlib.contextmanager
  def open(self, mode='r'):
    raise NotImplementedError

  def absolute(self):
    raise NotImplementedError

  def glob(self, pattern):
    raise NotImplementedError

  def exists(self):
    raise NotImplementedError

  def isfile(self):
    raise NotImplementedError

  def isdir(self):
    raise NotImplementedError

  def mkdirs(self):
    raise NotImplementedError

  def remove(self):
    raise NotImplementedError

  def rmtree(self):
    raise NotImplementedError

  def copy(self, dest):
    raise NotImplementedError


class LocalPath(Path):

  def __init__(self, path):
    super().__init__(os.path.expanduser(str(path)))

  @contextlib.contextmanager
  def open(self, mode='r'):
    with open(str(self), mode=mode) as f:
      yield f

  def absolute(self):
    return type(self)(os.path.absolute(str(self)))

  def glob(self, pattern):
    for path in glob.glob(f'{str(self)}/{pattern}'):
      yield type(self)(path)

  def exists(self):
    return os.path.exists(str(self))

  def isfile(self):
    return os.path.isfile(str(self))

  def isdir(self):
    return os.path.isdir(str(self))

  def mkdirs(self):
    os.makedirs(str(self), exist_ok=True)

  def remove(self):
    os.rmdir(str(self)) if self.isdir() else os.remove(str(self))

  def rmtree(self):
    shutil.rmtree(self)

  def copy(self, dest):
    shutil.copytree(self, type(self)(dest), dirs_exist_ok=True)


Path.filesystems = [
    # (GFilePath, lambda path: path.startswith('gs://')),
    # (GFilePath, lambda path: path.startswith('/cns/')),
    (LocalPath, lambda path: True),
]
