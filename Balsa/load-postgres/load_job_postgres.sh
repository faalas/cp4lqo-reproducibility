#!/bin/bash
# Copyright 2022 The Balsa Authors.
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

set -ex

DATA_DIR=$1
DBNAME=${2:-imdbload}

dropdb --if-exists $DBNAME
createdb $DBNAME

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
psql $DBNAME -f "$DIR/schema.sql"
psql $DBNAME -f "$DIR/fkindexes.sql"

pushd $DATA_DIR
psql $DBNAME -c "\copy name from 'name.csv' escape '\' csv header" &
psql $DBNAME -c "\copy char_name from 'char_name.csv' escape '\' csv header" &
psql $DBNAME -c "\copy comp_cast_type from 'comp_cast_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy company_name from 'company_name.csv' escape '\' csv header" &
psql $DBNAME -c "\copy company_type from 'company_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy info_type from 'info_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy keyword from 'keyword.csv' escape '\' csv header" &
psql $DBNAME -c "\copy kind_type from 'kind_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy link_type from 'link_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy role_type from 'role_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy title from 'title.csv' escape '\' csv header" &
psql $DBNAME -c "\copy aka_name from 'aka_name.csv' escape '\' csv header" &
psql $DBNAME -c "\copy aka_title from 'aka_title.csv' escape '\' csv header" &
psql $DBNAME -c "\copy cast_info from 'cast_info.csv' escape '\' csv header" &
psql $DBNAME -c "\copy complete_cast from 'complete_cast.csv' escape '\' csv header" &
psql $DBNAME -c "\copy movie_companies from 'movie_companies.csv' escape '\' csv header" &
psql $DBNAME -c "\copy movie_info from 'movie_info.csv' escape '\' csv header" &
psql $DBNAME -c "\copy movie_info_idx from 'movie_info_idx.csv' escape '\' csv header" &
psql $DBNAME -c "\copy movie_keyword from 'movie_keyword.csv' escape '\' csv header" &
psql $DBNAME -c "\copy movie_link from 'movie_link.csv' escape '\' csv header" &
psql $DBNAME -c "\copy person_info from 'person_info.csv' escape '\' csv header" &
wait
popd

# Declare FK constraints.
psql $DBNAME -f "$DIR/add_fks.sql"

# Create histograms.
psql $DBNAME -c "ANALYZE VERBOSE;"
