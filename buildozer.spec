[app]
# Nome visível do app
title = Teku

# Nome interno do pacote (não pode ter espaços)
package.name = gloweq

# Domínio invertido (podes mudar "org.example" para algo teu)
package.domain = org.example

# Diretório onde está o main.py
source.dir = .

# Extensões de ficheiros a incluir
source.include_exts = py,png,jpg,kv,txt

# Versão da app
version = 0.1

# Módulos Python a incluir
requirements = python3,kivy,pyjnius,numpy

# Permissões Android necessárias
android.permissions = RECORD_AUDIO

# (opcional) Ícone e banner
# icon.filename = %(source.dir)s/data/icon.png
# presplash.filename = %(source.dir)s/data/presplash.png

# Idioma do app (não obrigatório)
# fullscreen = 0

# Configurações de orientação
orientation = portrait

# API e NDK a usar
android.api = 33
android.ndk = 25b
android.minapi = 24

# Target Android SDK
android.sdk = 33

# Se quiseres compilar AAB (App Bundle) também:
# android.archs = armeabi-v7a, arm64-v8a
# android.release_artifact = aab

# Configuração de build
p4a.branch = master

# Nome do ficheiro final
package.version_code = 1

# (Opcional) Se quiseres ativar modo debug
# log_level = 2

# (Opcional) Para reduzir tamanho do APK
# android.strip = True

# (Opcional) Adicionar serviço de gravação ou outras libs
# requirements = python3,kivy,pyjnius,numpy,sdl2_ttf

[buildozer]
# Nível de logs (0=silencioso, 2=verbose)
log_level = 2

# Onde guardar ficheiros temporários e SDK/NDK
build_dir = .buildozer

# Se quiseres usar Docker em vez de instalar dependências localmente:
# use_docker = True
