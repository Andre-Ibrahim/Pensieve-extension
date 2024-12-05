import os

DASH_FILE = ['235k', '375k',
             '560k', '750k',
             '1050k', '1750k',
             '2350k', '3000k',
             '4300k']

for t in os.listdir('./'):
    if os.path.isdir(t):
        print(os.path.isdir(os.path.join(t, os.listdir(t)[1])))
        for j in os.listdir(t):
            if os.path.isdir(os.path.join(t, j)):
                print(os.listdir(os.path.join(t, j)))
                for i in os.listdir(os.path.join(t, j)):
                    if i == "size":
                        for q in os.listdir(os.path.join(t, j, i)):
                            print(q)
                            quality = q.split('_')[3]

                            index = DASH_FILE.index(quality)
                            filename = f"video_size_{index}"
                            # append or create a file with name filename
                            # and write the content of that filename to that new file
                            with open(filename, 'a') as nf:
                                with open(os.path.join(t, j, i, q), 'r') as f:
                                    nf.write(f.read())
