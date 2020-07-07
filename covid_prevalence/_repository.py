'''
Handle loading/saving data using Git repository
'''

import git

def gitpush(message, repo_path='/content/covid-prevalence'):
  repo = git.Repo(repo_path)
  repo.git.add('--all')
  repo.git.commit('-m', message, author='Steven Horn')
  repo.remotes.origin.push()
  #repo.remotes.gitlab.push()


