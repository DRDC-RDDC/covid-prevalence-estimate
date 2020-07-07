'''
Handle loading/saving data using Git repository
'''

import git

def gitpush(message):
  repo = git.Repo('/content/covid-prevalence')
  repo.git.add('--all')
  repo.git.commit('-m', message, author='Steven Horn')
  repo.remotes.gitlab.push()


