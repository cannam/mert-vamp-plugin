
project(
  'MERT Vamp Plugin',
  'cpp',
  # For a Vamp plugin the version must be a single integer
  version: '1',
  license: 'MIT',
  default_options: [
    'cpp_std=c++17',
    'warning_level=1',
    'buildtype=release',
    'b_ndebug=if-release',
  ],
)

conf = configuration_data()
conf.set('version', meson.project_version())
configure_file(input: 'version.h.in', output: 'version.h', configuration: conf)

system = host_machine.system()
architecture = host_machine.cpu_family()

libtorch_path = get_option('libtorch_path')
mkl_path = get_option('mkl_path')

config_summary = { 'System': system,
                   'Architecture': architecture,
                   'C++ Compiler': meson.get_compiler('cpp').get_id(),
                   'LibTorch Path': libtorch_path,
                   'MKL Path': mkl_path,
                 }

openmp_dep = dependency('openmp', required: false)

feature_defines = []
feature_libs = []
general_defines = []
general_link_args = []

if system == 'darwin'
  general_defines += [
    '-mmacosx-version-min=10.15',
    '-Wno-unknown-pragmas',
  ]
  general_link_args += [
    '-mmacosx-version-min=10.15',
  ]
endif

if libtorch_path != ''
  feature_defines += [
    '-DUSE_LIBTORCH',
    '-I' + libtorch_path / 'include',
    '-I' + libtorch_path / 'include/torch/csrc',
    '-I' + libtorch_path / 'include/torch/csrc/api/include',
  ]
  feature_libs += [
    '-L' + libtorch_path / 'lib',
    '-Wl,--rpath', libtorch_path / 'lib',
    '-ltorch', '-ltorch_cpu', '-lc10',
  ]
  config_summary += {
    'Computation': 'Torch',
    'Threading': 'Torch',
  }
elif system == 'darwin'
  feature_defines += [
    '-DUSE_CBLAS',
    '-DACCELERATE_NEW_LAPACK=1',
    '-DUSE_DISPATCH',
  ]
  feature_libs += [
    '-framework', 'Accelerate',
  ]
  config_summary += {
    'Computation': 'Accelerate',
    'Threading': 'Dispatch',
  }
else
  if openmp_dep.found()
    feature_defines += [
      '-fopenmp',
    ]
    feature_libs += [
      '-fopenmp',
    ]
    config_summary += {
      'Threading': 'OpenMP',
    }
  else
    config_summary += {
      'Threading': '(none)'
    }
  endif # end openmp_dep
  
  if mkl_path != ''
    feature_defines += [
      '-DUSE_CBLAS',
      '-DUSE_MKL',
      '-I' + mkl_path / 'include',
      '-I' + mkl_path / 'include/mkl'
    ]
    feature_libs += [
      '-L' + (mkl_path / 'lib'),
      '-Wl,-Bstatic',
      '-lmkl_intel_lp64', '-lmkl_gnu_thread', '-lmkl_core',
      '-Wl,-Bdynamic',
    ]
    config_summary += {
      'Computation': 'MKL',
    }
  else
    config_summary += {
      'Matrix Library': '(none)'
    }
  endif # end not mkl_path
  
endif # end not darwin
  
plugin_files = [
  'MERTVampPlugin.cpp',
  'plugins.cpp',
  'data/weights.cpp',
]

qmdsp_files = [
  'ext/qm-dsp/dsp/rateconversion/Resampler.cpp',
  'ext/qm-dsp/maths/MathUtilities.cpp',
  'ext/qm-dsp/base/KaiserWindow.cpp',
  'ext/qm-dsp/base/SincWindow.cpp',
]

weight_files = [
  'data/weights_00.cpp', 'data/weights_01.cpp', 'data/weights_02.cpp',
  'data/weights_03.cpp', 'data/weights_04.cpp', 'data/weights_05.cpp',
  'data/weights_06.cpp', 'data/weights_07.cpp', 'data/weights_08.cpp',
  'data/weights_09.cpp', 'data/weights_10.cpp', 'data/weights_11.cpp',
  'data/weights_12.cpp', 'data/weights_13.cpp', 'data/weights_14.cpp',
  'data/weights_15.cpp', 'data/weights_16.cpp', 'data/weights_17.cpp',
  'data/weights_18.cpp', 'data/weights_19.cpp', 'data/weights_20.cpp',
  'data/weights_21.cpp', 'data/weights_22.cpp', 'data/weights_23.cpp',
  'data/weights_24.cpp', 'data/weights_25.cpp', 'data/weights_26.cpp',
  'data/weights_27.cpp', 'data/weights_28.cpp', 'data/weights_29.cpp',
  'data/weights_30.cpp', 'data/weights_31.cpp', 'data/weights_32.cpp',
  'data/weights_33.cpp', 'data/weights_34.cpp', 'data/weights_35.cpp',
  'data/weights_36.cpp', 'data/weights_37.cpp', 'data/weights_38.cpp',
]

vamp_sdk_files = [
  'ext/vamp-plugin-sdk/src/vamp-sdk/FFT.cpp',
  'ext/vamp-plugin-sdk/src/vamp-sdk/PluginAdapter.cpp',
  'ext/vamp-plugin-sdk/src/vamp-sdk/RealTime.cpp',
]

vamp_symbol_args = []

if system == 'linux'
  vamp_symbol_args += [
    '-Wl,--version-script=' + meson.current_source_dir() / 'vamp-plugin.map'
  ]
  vamp_install_dir = '/usr/local/lib/vamp'
elif system == 'darwin'
  vamp_symbol_args += [
    '-exported_symbols_list', meson.current_source_dir() / 'vamp-plugin.list'
  ]
  vamp_install_dir = '/Library/Audio/Plug-Ins/Vamp'
elif system == 'windows'
  vamp_symbol_args += [
    '-EXPORT:vampGetPluginDescriptor'
  ]
  vamp_install_dir = 'C:/Program Files/Vamp Plugins'
endif

install_data(
  'mert-vamp-plugin.cat',
  install_dir: vamp_install_dir,
)

weight_archive = static_library(
  'weights',
  weight_files,
  cpp_args: [
    general_defines,
  ],
)
  
plugin_library = shared_library(
  'mert-vamp-plugin',
  plugin_files,
  qmdsp_files,
  vamp_sdk_files,
  cpp_args: [
    feature_defines,
    general_defines,
  ],
  dependencies: [
    openmp_dep,
  ],
  include_directories: [
    'ext/vamp-plugin-sdk',
    'ext/qm-dsp',
  ],
  link_with: [
    weight_archive,
  ],
  link_args: [
    vamp_symbol_args,
    general_link_args,
    feature_libs,
  ],
  name_prefix: '',
  install: true,
  install_dir: vamp_install_dir,
)  

summary(config_summary, section: 'Configuration', bool_yn: true)
