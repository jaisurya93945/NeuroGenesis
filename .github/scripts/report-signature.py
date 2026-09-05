#!/usr/bin/env python3
"""Print whether the commit just published came back signed."""
import json
import os
import sys
import urllib.request

sha = os.environ.get('PUBLISHED_SHA')
if not sha:
    print('no commit was made this run — nothing to report')
    sys.exit(0)

url = 'https://api.github.com/repos/%s/commits/%s' % (os.environ['GITHUB_REPOSITORY'], sha)
req = urllib.request.Request(url, headers={
    'Authorization': 'bearer %s' % os.environ['GH_TOKEN'],
    'User-Agent': 'neurogenesis-publish',
})
with urllib.request.urlopen(req, timeout=30) as r:
    d = json.load(r)

c = d['commit']
v = c.get('verification', {})
line = '=' * 58
print(line)
print('commit    : %s' % d['sha'][:12])
print('author    : %s <%s>' % (c['author']['name'], c['author']['email']))
print('committer : %s <%s>' % (c['committer']['name'], c['committer']['email']))
print('VERIFIED  : %s' % v.get('verified'))
print('reason    : %s' % v.get('reason'))
print(line)

if not v.get('verified'):
    sys.exit('commit is NOT verified (%s)' % v.get('reason'))

if 'bot' in c['author']['email']:
    print('Authored by the Actions bot. Add a RESUME_PAT secret to author as yourself.')
else:
    print('Signed and authored by you.')
