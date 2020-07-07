'''
Handle loading/saving data using Git repository
'''

import git
import logging

log = logging.getLogger(__name__)

def gitpush(message, repo_path='/content/covid-prevalence', repo_branch='latest'):
  repo = git.Repo(repo_path)
  repo.git.add('--all')
  repo.git.commit('-m', message, author='Steven Horn')
  log.info('Pushing branch %s to origin' % repo_branch)
  repo.remotes.origin.push(repo_branch)
  #repo.remotes.gitlab.push()