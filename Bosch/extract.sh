#!/bin/bash

# Script pour extraire un certain nombre de lignes des fichiers téléchargés.
# utilise le fichier lines_sort pour avoir la même base des lignes entre nous
# lines_sort peut être généré via un autre script

sed -n -f lines_sort $1 > $2 
