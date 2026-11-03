def plot_upa(TH, PH, gain, title, step=4):
    THs = TH[::step, ::step]
    PHs = PH[::step, ::step]
    R = gain[::step, ::step]
    R = R / R.max()
    X = R * np.sin(THs) * np.cos(PHs)
    Y = R * np.sin(THs) * np.sin(PHs)
    Z = R * np.cos(THs)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0,
                    antialiased=False, shade=True)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()
