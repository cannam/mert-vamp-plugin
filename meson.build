
project(
  'MERT Vamp Plugin',
  'cpp',
  # For a Vamp plugin the version must be a single integer
  version: '1',
  license: 'MIT',
  default_options: [
    'cpp_std=c++17',
    'warning_level=2',
    'buildtype=release',
    'b_ndebug=if-release',
  ],
)

conf = configuration_data()
conf.set('version', meson.project_version())
configure_file(input: 'version.h.in', output: 'version.h', configuration: conf)

system = host_machine.system()

plugin_files = [
  'MERTVampPlugin.cpp',
  'plugins.cpp',
  'data/weights.cpp',
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
)
  
plugin_library = shared_library(
  'mert-vamp-plugin',
  plugin_files,
  vamp_sdk_files,
  include_directories: [
    'ext/vamp-plugin-sdk',
  ],
  link_with: [
    weight_archive,
  ],
  link_args: [
    vamp_symbol_args,
  ],
  name_prefix: '',
  install: true,
  install_dir: vamp_install_dir,
)  
