#!/usr/bin/env python
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Hybrid Windows Crosstool wrapper: clang-cl (host C/C++) + nvcc/cl.exe (CUDA)."""

from argparse import ArgumentParser
import locale
import os
import re
import shutil
import subprocess
import sys
import tempfile

if hasattr(sys.stdout, 'reconfigure'):
  sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Template values set by cuda_configure.bzl
CPU_COMPILER = '%{cpu_compiler}'
MSVC_CL_PATH = '%{msvc_cl_exe}'
LLVM_ROOT = '%{llvm_root}'
IS_CLANG_CL = ('clang-cl' in os.path.basename(CPU_COMPILER).lower())

NVCC_PATH = '%{nvcc_path}'
NVCC_VERSION = '%{cuda_version}'
NVCC_TEMP_DIR = '%{nvcc_tmp_dir}'

_LLVM_ROOT_NORM = (
    os.path.normcase(os.path.normpath(LLVM_ROOT)) if LLVM_ROOT else ''
)
_LLVM_ROOT_PREFIX = (
    (_LLVM_ROOT_NORM if _LLVM_ROOT_NORM.endswith(os.sep)
     else _LLVM_ROOT_NORM + os.sep)
    if _LLVM_ROOT_NORM else ''
)


def _IsUnderLlvmRoot(path):
  """Returns True if path is inside the configured LLVM_ROOT directory."""
  if not _LLVM_ROOT_NORM:
    return False
  norm_path = os.path.normcase(os.path.normpath(path.strip().strip('"')))
  return (
      norm_path == _LLVM_ROOT_NORM
      or norm_path.startswith(_LLVM_ROOT_PREFIX)
  )


def ExpandParamFiles(argv):
  """Expands Bazel @params files into argv entries."""
  expanded = []
  for arg in argv:
    if arg.startswith('@') and os.path.exists(arg[1:]):
      with open(arg[1:], 'r', encoding='utf-8', errors='replace') as params_file:
        for line in params_file:
          line = line.strip()
          if line:
            line = line.replace(r'\"', '"')
            expanded.append(line)
    else:
      expanded.append(arg)
  return expanded


def StripBazelWrapperFlags(argv):
  """Drops -B wrapper flags before invoking nvcc, cl.exe, or clang-cl."""
  stripped = []
  skip_next = False
  for i, arg in enumerate(argv):
    if skip_next:
      skip_next = False
      continue
    if (arg == '-B' and i + 1 < len(argv) and
        'msvc_wrapper_for_nvcc.py' in argv[i + 1].replace('\\', '/')):
      skip_next = True
      continue
    stripped.append(arg)
  return stripped


_STD_MAP = {
    'c11': 'c11',
    'gnu11': 'c11',
    'c17': 'c17',
    'gnu17': 'c17',
    'c++14': 'c++14',
    'gnu++14': 'c++14',
    'c++17': 'c++17',
    'gnu++17': 'c++17',
    'c++20': 'c++20',
    'gnu++20': 'c++20',
    'c++latest': 'c++latest',
}


def FilterForMsvcHost(argv):
  """Translates/drops clang-cl and GCC flags that MSVC cl.exe rejects under nvcc."""
  ignored_prefixes = (
      '-Wno-',
      '/Wno-',
      '-Wno-error=',
      '/Wno-error=',
      '-Werror=',
      '/Werror=',
      '-W',
      '/clang:',
      '-Qunused-arguments',
      '-m',
      '-f',
      '--cuda-feature=',
      '/d2ReducedOptimizeHugeFunctions',
  )
  filtered = []
  has_zc_preprocessor = False
  for arg in argv:
    if arg.startswith('/clang:-march='):
      # Keep nvcc's MSVC host stub at baseline SSE2 so cudafe++ does not parse
      # AVX2/AVX-512 intrinsics headers (clang-cl still compiles all host C++ with target arch).
      continue
    if arg == '/Zc:preprocessor':
      has_zc_preprocessor = True
      filtered.append(arg)
      continue
    if arg.startswith(ignored_prefixes):
      continue
    if arg.startswith(('-std=', '-std:')):
      std_val = arg[5:]
      if std_val in _STD_MAP:
        filtered.append('/std:' + _STD_MAP[std_val])
      continue
    # Avoid passing clang-only warning suppressions or /Zm that warn in newer MSVC
    if arg in ('/D__PRFCHWINTRIN_H',):
      continue
    filtered.append(arg)
  if not has_zc_preprocessor:
    filtered.append('/Zc:preprocessor')
  return filtered


def FilterForClangCl(argv):
  """Cleans flags passed to clang-cl.exe for non-CUDA host C/C++ files."""
  filtered = ['-Qunused-arguments']
  for arg in argv:
    if arg in ('/Zm500', '/Zc:preprocessor', '/d2ReducedOptimizeHugeFunctions'):
      continue
    if arg.startswith(('-std=', '-std:')):
      std_val = arg[5:]
      if std_val in _STD_MAP:
        filtered.append('/std:' + _STD_MAP[std_val])
        continue
    filtered.append(arg)
  return filtered


def FilterForMsvcDirect(argv):
  """Drops GCC/Clang warning flags that MSVC rejects when cl.exe is CPU_COMPILER."""
  ignored_prefixes = (
      '-Wno-',
      '/Wno-',
      '-Wno-error=',
      '/Wno-error=',
      '-Werror=',
      '/Werror=',
      '/clang:',
      '-Qunused-arguments',
  )
  filtered = []
  for arg in argv:
    if arg.startswith(ignored_prefixes):
      continue
    if arg.startswith(('-std=', '-std:')):
      std_val = arg[5:]
      if std_val in _STD_MAP:
        mapped_std = _STD_MAP[std_val]
        filtered.append('/std:' + mapped_std)
        if mapped_std in ('c11', 'c17'):
          filtered.append('/experimental:c11atomics')
        continue
    filtered.append(arg)
  return filtered


def Log(s):
  print('gpus/crosstool: {0}'.format(s))


def DecodeCompilerOutput(data):
  """Decodes compiler output using UTF-8 or the local Windows code page."""
  encodings = [
      'utf-8',
      locale.getpreferredencoding(False),
      'mbcs',
      sys.getfilesystemencoding(),
  ]
  tried = set()
  for encoding in encodings:
    if not encoding or encoding in tried:
      continue
    tried.add(encoding)
    try:
      return data.decode(encoding)
    except (LookupError, UnicodeDecodeError):
      pass
  return data.decode('utf-8', errors='replace')


def EscapeMsvcResponseArg(arg):
  """Escapes macro string-literal quotes and spaces for response files."""
  if arg.startswith(('/D', '-D')):
    arg = arg.replace('"', r'\"')
  if ' ' in arg and not (arg.startswith('"') and arg.endswith('"')):
    return '"' + arg + '"'
  return arg


def WriteResponseFile(args):
  """Writes compiler args to a temporary response file and returns @file."""
  response_file = tempfile.NamedTemporaryFile(
      mode='w', suffix='.rsp', prefix='cc_args_', delete=False, encoding='utf-8')
  try:
    response_file.write('\n'.join(EscapeMsvcResponseArg(arg) for arg in args))
    response_file.write('\n')
    return response_file.name
  finally:
    response_file.close()


def RunCompilerCommand(cmd, env=None):
  """Runs a compiler and emits Bazel-compatible output."""
  if env is None:
    env = os.environ.copy()
  env.setdefault('VSLANG', '1033')
  proc = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
  output, _ = proc.communicate()
  if output:
    sys.stdout.write(DecodeCompilerOutput(output))
    sys.stdout.flush()
  return proc.returncode


def GetOptionValue(argv, option):
  parser = ArgumentParser(prefix_chars='-/')
  parser.add_argument(option, nargs='*', action='append')
  option_attr = option.lstrip('-/').replace('-', '_')
  args, leftover = parser.parse_known_args(argv)
  if args and vars(args)[option_attr]:
    return (sum(vars(args)[option_attr], []), leftover)
  return ([], leftover)


def _update_options(nvcc_options):
  update_options = {'relaxed-constexpr': 'expt-relaxed-constexpr'}
  return [update_options.get(opt, opt) for opt in nvcc_options]


def GetNvccOptions(argv):
  parser = ArgumentParser()
  parser.add_argument('-nvcc_options', nargs='*', action='append')
  args, leftover = parser.parse_known_args(argv)
  if args.nvcc_options:
    options = _update_options(sum(args.nvcc_options, []))
    return (['--' + a for a in options], leftover)
  return ([], leftover)


def _QuoteNvccPath(path, always_quote=False):
  """Normalizes and quotes a path for nvcc --options-file."""
  clean = path.strip().strip('"').replace('\\', '/')
  if re.match(r'^[A-Za-z]:/+$', clean):
    clean = clean[:3]
  elif clean != '/':
    clean = clean.rstrip('/')
  if always_quote or ' ' in clean or '\t' in clean:
    return '"' + clean + '"'
  return clean


def InvokeNvcc(argv, log=False):
  """Invoke NVIDIA nvcc using MSVC cl.exe as host compiler for .cu/.cu.cc files."""
  src_files = [
      f for f in argv
      if re.search(r'\.cpp$|\.cc$|\.c$|\.cxx$|\.C$|\.cu$', f)
  ]
  if not src_files:
    raise RuntimeError('No source files found for cuda compilation.')

  out_file = [f for f in argv if f.startswith('/Fo')]
  if len(out_file) != 1:
    raise RuntimeError('Please specify exactly one /Fo output file for cuda compilation.')
  out = ['-o', _QuoteNvccPath(out_file[0][len('/Fo'):])]

  nvcc_compiler_options, argv = GetNvccOptions(argv)

  opt_option, argv = GetOptionValue(argv, '/O')
  opt = ['-g']
  if opt_option and opt_option[0] != 'd':
    opt = ['-O2']

  include_options, argv = GetOptionValue(argv, '/I')
  includes = ['-I ' + _QuoteNvccPath(inc) for inc in include_options]

  defines, argv = GetOptionValue(argv, '/D')
  defines = ['-D' + d for d in defines]

  undefines, argv = GetOptionValue(argv, '/U')
  undefines = ['-U' + u for u in undefines]

  fatbin_options, argv = GetOptionValue(argv, '-Xcuda-fatbinary')
  fatbin_options = ['--fatbin-options=' + option for option in fatbin_options]

  m_options = ['-m64']

  nvccopts = ['-D_FORCE_INLINES']
  if MSVC_CL_PATH and os.path.exists(MSVC_CL_PATH):
    nvccopts.append(
        '--compiler-bindir=' +
        _QuoteNvccPath(os.path.dirname(MSVC_CL_PATH), always_quote=True)
    )

  compute_capabilities, argv = GetOptionValue(argv, '--cuda-gpu-arch')
  for capability in compute_capabilities:
    cap = capability[len('sm_'):]
    nvccopts.append(r'-gencode=arch=compute_%s,"code=sm_%s"' % (cap, cap))

  compute_capabilities, argv = GetOptionValue(argv, '--cuda-include-ptx')
  for capability in compute_capabilities:
    cap = capability[len('sm_'):]
    nvccopts.append(r'-gencode=arch=compute_%s,"code=compute_%s"' % (cap, cap))

  _, argv = GetOptionValue(argv, '--no-cuda-include-ptx')

  host_compiler_options = FilterForMsvcHost(
      [option for option in argv if option not in (src_files + out_file)]
  )

  # Pass MSVC/Windows SDK INCLUDE paths explicitly to nvcc
  if 'INCLUDE' in os.environ:
    for p in os.environ['INCLUDE'].split(';'):
      p = p.strip()
      if p and not _IsUnderLlvmRoot(p):
        nvccopts.append('--system-include=' + _QuoteNvccPath(p, always_quote=True))
  if 'LIB' in os.environ:
    for p in os.environ['LIB'].split(';'):
      p = p.strip()
      if p and not _IsUnderLlvmRoot(p):
        nvccopts.append('--library-path=' + _QuoteNvccPath(p, always_quote=True))

  nvccopts += nvcc_compiler_options
  nvccopts += undefines
  nvccopts += defines
  nvccopts += m_options
  nvccopts += fatbin_options
  if host_compiler_options:
    nvccopts.append('--compiler-options=' + ','.join(host_compiler_options))
  nvccopts += (
      ['-x', 'cu'] + opt + includes + out + ['-c'] +
      [_QuoteNvccPath(f) for f in src_files]
  )

  if os.path.isfile(NVCC_TEMP_DIR):
    os.remove(NVCC_TEMP_DIR)
  os.makedirs(NVCC_TEMP_DIR, exist_ok=True)
  keep_temp = '--keep' in nvcc_compiler_options
  tempdir = tempfile.mkdtemp(dir=NVCC_TEMP_DIR)
  try:
    if not keep_temp:
      nvccopts.append('--keep')
    if not any(
        opt_flag == '--keep-dir' or opt_flag.startswith('--keep-dir=')
        for opt_flag in nvcc_compiler_options
    ):
      nvccopts.append('--keep-dir=' + _QuoteNvccPath(tempdir, always_quote=True))
    nvccopts += ['--std c++17']
    if log:
      Log([NVCC_PATH] + nvccopts)

    with tempfile.NamedTemporaryFile(
        mode='w', dir=tempdir, delete=False, encoding='utf-8') as optsfile:
      optsfile.write('\n'.join(nvccopts))
      optsfile_name = optsfile.name

    nvcc_env = os.environ.copy()
    if 'INCLUDE' in nvcc_env:
      nvcc_env['INCLUDE'] = ';'.join(
          p for p in nvcc_env['INCLUDE'].split(';')
          if p.strip() and not _IsUnderLlvmRoot(p)
      )
    if 'LIB' in nvcc_env:
      nvcc_env['LIB'] = ';'.join(
          p for p in nvcc_env['LIB'].split(';')
          if p.strip() and not _IsUnderLlvmRoot(p)
      )
    return RunCompilerCommand(
        [NVCC_PATH, '--options-file', optsfile_name], env=nvcc_env)
  finally:
    if not keep_temp:
      shutil.rmtree(tempdir, ignore_errors=True)


def main():
  argv = StripBazelWrapperFlags(ExpandParamFiles(sys.argv[1:]))
  parser = ArgumentParser()
  parser.add_argument('-x', nargs=1)
  parser.add_argument('--cuda_log', action='store_true')
  args, leftover = parser.parse_known_args(argv)

  if args.x and args.x[0] == 'cuda':
    if args.cuda_log:
      Log('-x cuda -> nvcc + MSVC cl.exe')
    return InvokeNvcc(leftover, log=args.cuda_log)

  cpu_compiler_flags = [
      flag for flag in argv
      if not flag.startswith('--cuda_log')
      and not flag.startswith('-nvcc_options')
  ]

  if IS_CLANG_CL:
    cpu_compiler_flags = FilterForClangCl(cpu_compiler_flags)
  else:
    cpu_compiler_flags = FilterForMsvcDirect(cpu_compiler_flags)

  rsp_path = WriteResponseFile(cpu_compiler_flags)
  try:
    return RunCompilerCommand([CPU_COMPILER, '@' + rsp_path])
  finally:
    try:
      os.remove(rsp_path)
    except OSError:
      pass


if __name__ == '__main__':
  sys.exit(main())
