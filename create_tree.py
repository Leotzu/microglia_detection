from subprocess import call

command = 'tree '
command += '--dirsfirst '
command += '--noreport '
command += '-I '
command += "'location*|src|tree*|*.json|*tif*|*.zarr|*.n5|*check*|*log*|*snap*|*.md|*cache*"

for i in range(0,100):
    command += f'|{i}'
    command += f'|{i}.*'

command += "' "
command += '| '
command += "sed '1s/^/```/;$s/$/```/' "
command += '>> README.md'

call(command, shell=True)
