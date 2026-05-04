import os
import re
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from src.load_data import load_data
from src.features import features
from src.user_profiles import user_profiles
from src.retrieval import retrieval


print("Loading recommender...")

loader = load_data()
loader.preprocess_all_columns()

feat = features(loader)
profile = user_profiles(loader)
ret = retrieval(loader, feat, profile)

feat.build_vocabulary()
feat.train_word2vec()
feat.compute_IDF(loader.dataset["description"], field="description")
feat.compute_IDF(loader.dataset["genres"], field="genres")
feat.compute_IDF(loader.dataset["production_countries"], field="country")

ret.precompute_doc_matrices()

print("Recommender ready.")


class NetflixUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Netflix NLP Recommender")
        self.root.geometry("1250x750")
        self.root.configure(bg="#141414")

        self.users = loader.user_dataset["UserID"].tolist()
        self.current_user = self.users[0]
        self.avatar_choices = ["🙂", "😎", "🤖", "🍿", "🎬", "⭐"]

        self.poster_cache = {}
        self.pic_folder = os.path.join(os.path.dirname(__file__), "pics")

        self.make_top_bar()
        self.make_hero()
        self.make_cards_area()
        self.update_recommendations()

    def get_avatar(self, user_id):
        index = self.users.index(user_id) % len(self.avatar_choices)
        return self.avatar_choices[index]

    def clean_title_for_file(self, title):
        title = title.lower()
        title = title.replace("&", "")
        title = re.sub(r"[^a-z0-9]+", "_", title)
        title = title.strip("_")
        return title

    def get_poster_path(self, title):
        manual_names = {
            "Invader ZIM: Enter the Florpus": "invader_zim_enter_the_florpus.jpg",
            "Spy Kids: Mission Critical": "spykids_mission_critical.webp",
            "Daybreak": "daybreak.webp",
            "See You Yesterday": "see_you_yesterday.webp",
            "Gotham": "gotham.webp",
            "Love, Death & Robots": "love_death_robots.webp",
            "Masters of the Universe: Revelation": "masters_of_the_universe.webp",
            "Looper": "looper.webp",
            "iZombie": "izombie.webp",
            "Sleight": "sleight.webp",
        }

        if title in manual_names:
            path = os.path.join(self.pic_folder, manual_names[title])
            if os.path.exists(path):
                return path

        clean_name = self.clean_title_for_file(title)
        extensions = [".jpg", ".jpeg", ".png", ".webp"]

        for ext in extensions:
            path = os.path.join(self.pic_folder, clean_name + ext)
            if os.path.exists(path):
                return path

        return None

    def load_poster_image(self, title, width=210, height=310):
        path = self.get_poster_path(title)

        if path is None:
            return None

        key = title + str(width) + str(height)

        if key in self.poster_cache:
            return self.poster_cache[key]

        image = Image.open(path)
        image = image.resize((width, height))
        photo = ImageTk.PhotoImage(image)

        self.poster_cache[key] = photo
        return photo

    def make_top_bar(self):
        top = tk.Frame(self.root, bg="#141414")
        top.pack(fill="x", padx=35, pady=(20, 10))

        logo = tk.Label(
            top,
            text="NETFLIX",
            font=("Arial", 34, "bold"),
            fg="#E50914",
            bg="#141414"
        )
        logo.pack(side="left")

        nav = tk.Label(
            top,
            text="Home     Recommender     Movies     My List",
            font=("Arial", 12),
            fg="white",
            bg="#141414"
        )
        nav.pack(side="left", padx=35)

        self.profile_button = tk.Label(
            top,
            text=self.get_avatar(self.current_user) + " ▾",
            font=("Arial", 20),
            fg="white",
            bg="#141414",
            cursor="hand2"
        )
        self.profile_button.pack(side="right")

        self.profile_button.bind("<Button-1>", lambda event: self.toggle_profile_menu())

    def toggle_profile_menu(self):
        if hasattr(self, "profile_menu") and self.profile_menu.winfo_exists():
            self.profile_menu.destroy()
            return

        self.profile_menu = tk.Toplevel(self.root)
        self.profile_menu.overrideredirect(True)
        self.profile_menu.configure(bg="#181818")

        x = self.root.winfo_x() + self.root.winfo_width() - 260
        y = self.root.winfo_y() + 85
        self.profile_menu.geometry(f"220x300+{x}+{y}")

        title = tk.Label(
            self.profile_menu,
            text="Who's watching?",
            font=("Arial", 13, "bold"),
            fg="white",
            bg="#181818"
        )
        title.pack(anchor="w", padx=15, pady=(15, 10))

        for user_id in self.users[:10]:
            row = tk.Frame(self.profile_menu, bg="#181818", cursor="hand2")
            row.pack(fill="x", padx=10, pady=4)

            avatar = tk.Label(
                row,
                text=self.get_avatar(user_id),
                font=("Arial", 18),
                fg="white",
                bg="#181818"
            )
            avatar.pack(side="left", padx=8)

            name = tk.Label(
                row,
                text=user_id,
                font=("Arial", 11),
                fg="white",
                bg="#181818"
            )
            name.pack(side="left")

            for w in [row, avatar, name]:
                w.bind("<Button-1>", lambda event, u=user_id: self.switch_profile(u))

    def switch_profile(self, user_id):
        self.current_user = user_id

        if hasattr(self, "profile_menu") and self.profile_menu.winfo_exists():
            self.profile_menu.destroy()

        self.profile_button.config(text=self.get_avatar(self.current_user) + " ▾")
        self.update_recommendations()

    def make_hero(self):
        hero = tk.Frame(self.root, bg="#181818", height=160)
        hero.pack(fill="x", padx=35, pady=(10, 25))
        hero.pack_propagate(False)

        left = tk.Frame(hero, bg="#181818")
        left.pack(side="left", fill="both", expand=True, padx=25, pady=20)

        title = tk.Label(
            left,
            text="NLP Movie Recommender",
            font=("Arial", 30, "bold"),
            fg="white",
            bg="#181818"
        )
        title.pack(anchor="w")

        desc = tk.Label(
            left,
            text="Personalized movie recommendations generated from your Netflix-style user profile.",
            font=("Arial", 13),
            fg="#b3b3b3",
            bg="#181818",
            wraplength=600,
            justify="left"
        )
        desc.pack(anchor="w", pady=(10, 20))

        self.user_display_label = tk.Label(
            left,
            text=f"Current Profile: {self.current_user}",
            font=("Arial", 12, "bold"),
            fg="#E50914",
            bg="#181818"
        )
        self.user_display_label.pack(anchor="w")

        right = tk.Label(
            hero,
            text="Because you watched...",
            font=("Arial", 18, "bold"),
            fg="#E50914",
            bg="#181818"
        )
        right.pack(side="right", padx=40)

    def make_cards_area(self):
        self.section_title = tk.Label(
            self.root,
            text="Recommended For You",
            font=("Arial", 22, "bold"),
            fg="white",
            bg="#141414"
        )
        self.section_title.pack(anchor="w", padx=35, pady=(0, 10))

        container = tk.Frame(self.root, bg="#141414")
        container.pack(fill="both", expand=True, padx=35, pady=(0, 20))

        self.canvas = tk.Canvas(container, bg="#141414", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.cards_frame = tk.Frame(self.canvas, bg="#141414")
        self.canvas.create_window((0, 0), window=self.cards_frame, anchor="nw")

        self.cards_frame.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.bind_all(
            "<MouseWheel>",
            lambda event: self.canvas.yview_scroll(
                int(-1 * (event.delta / 120)),
                "units"
            )
        )

    def update_recommendations(self):
        for widget in self.cards_frame.winfo_children():
            widget.destroy()

        self.user_display_label.config(text=f"Current Profile: {self.current_user}")

        recs = ret.recommend_for_user(self.current_user, k=10)

        row = 0
        col = 0

        for _, movie in recs.iterrows():
            title = movie["title"]
            description = movie["description"]
            score = movie["score"]

            card = tk.Frame(
                self.cards_frame,
                bg="#1f1f1f",
                width=210,
                height=395,
                highlightbackground="#2a2a2a",
                highlightthickness=1
            )
            card.grid(row=row, column=col, padx=10, pady=15)
            card.grid_propagate(False)

            poster_image = self.load_poster_image(title)

            if poster_image is not None:
                poster = tk.Label(
                    card,
                    image=poster_image,
                    bg="#303030"
                )
                poster.image = poster_image
            else:
                poster = tk.Label(
                    card,
                    text=title,
                    font=("Arial", 13, "bold"),
                    bg="#303030",
                    fg="#E50914",
                    wraplength=175,
                    justify="center"
                )

            poster.pack(fill="x")

            info = tk.Frame(card, bg="#1f1f1f")
            info.pack(fill="both", expand=True, padx=12, pady=10)

            match = tk.Label(
                info,
                text=f"{int(score * 100)}% Match",
                font=("Arial", 10, "bold"),
                bg="#1f1f1f",
                fg="#46d369"
            )
            match.pack(anchor="w")

            meta = tk.Label(
                info,
                text="NLP Recommendation",
                font=("Arial", 9),
                bg="#1f1f1f",
                fg="#b3b3b3"
            )
            meta.pack(anchor="w", pady=(4, 0))

            click = tk.Label(
                info,
                text="Click for details",
                font=("Arial", 9, "bold"),
                bg="#1f1f1f",
                fg="white"
            )
            click.pack(anchor="w", pady=(12, 0))

            for w in [card, poster, info, match, meta, click]:
                w.bind(
                    "<Button-1>",
                    lambda event, t=title, d=description, s=score:
                    self.show_details(t, d, s)
                )

            col += 1
            if col == 5:
                col = 0
                row += 1

    def show_details(self, title, description, score):
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("750x520")
        popup.configure(bg="#141414")

        poster_image = self.load_poster_image(title, width=230, height=340)

        left = tk.Frame(popup, bg="#141414")
        left.pack(side="left", padx=30, pady=30)

        if poster_image is not None:
            poster = tk.Label(left, image=poster_image, bg="#141414")
            poster.image = poster_image
            poster.pack()
        else:
            poster = tk.Label(
                left,
                text=title,
                font=("Arial", 18, "bold"),
                fg="#E50914",
                bg="#303030",
                wraplength=200,
                width=16,
                height=10
            )
            poster.pack()

        right = tk.Frame(popup, bg="#141414")
        right.pack(side="left", fill="both", expand=True, padx=10, pady=30)

        title_label = tk.Label(
            right,
            text=title,
            font=("Arial", 24, "bold"),
            fg="#E50914",
            bg="#141414",
            wraplength=430,
            justify="left"
        )
        title_label.pack(anchor="w")

        score_label = tk.Label(
            right,
            text=f"{int(score * 100)}% Match    Score: {score:.4f}",
            font=("Arial", 12, "bold"),
            fg="#46d369",
            bg="#141414"
        )
        score_label.pack(anchor="w", pady=(15, 15))

        desc_label = tk.Label(
            right,
            text=description,
            font=("Arial", 12),
            fg="white",
            bg="#141414",
            wraplength=430,
            justify="left"
        )
        desc_label.pack(anchor="w")

        close_button = tk.Button(
            right,
            text="Close",
            font=("Arial", 11, "bold"),
            bg="#E50914",
            fg="white",
            relief="flat",
            command=popup.destroy
        )
        close_button.pack(anchor="w", pady=25, ipadx=25, ipady=6)


root = tk.Tk()
app = NetflixUI(root)
root.mainloop()