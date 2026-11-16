# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import tempfile

# Keep test shells out of the user's real ~/.ipython history and profile.
os.environ["IPYTHONDIR"] = tempfile.mkdtemp(prefix="aipython-test-ipython-")
