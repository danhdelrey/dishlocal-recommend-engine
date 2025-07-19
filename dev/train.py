import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from lightfm import LightFM
from lightfm.data import Dataset
import math

# =================================================================================
# SCRIPT HUẤN LUYỆN MODEL GỢI Ý - PHIÊN BẢN V6.0 (SENTIMENT-AWARE)
# - Phân biệt và học từ cả tương tác Tích cực và Tiêu cực.
# - Xử lý các trường hợp review mâu thuẫn.
# - Giữ lại tất cả các đặc trưng để xây dựng hồ sơ người dùng hoàn chỉnh.
# =================================================================================


# --- PHẦN 1: KẾT NỐI VÀ CÁC HÀM TIỆN ÍCH ---
def get_supabase_client() -> Client:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in .env file")
    return create_client(url, key)

# --- PHẦN 2: TRÍCH XUẤT VÀ XỬ LÝ DỮ LIỆU (LOGIC MỚI) ---
def fetch_data(supabase: Client):
    print("Fetching all data from Supabase...")
    
    # Lấy dữ liệu chính
    all_users_res = supabase.table('profiles').select('id').execute()
    all_posts_res = supabase.table('posts').select('id, food_category, author_id').execute()
    
    # Lấy TẤT CẢ các loại tương tác
    views_res = supabase.table('post_views').select('user_id, post_id, view_duration_ms').execute()
    likes_res = supabase.table('post_likes').select('user_id, post_id').execute()
    saves_res = supabase.table('post_saves').select('user_id, post_id').execute()
    # Lấy TẤT CẢ reviews, không lọc theo rating nữa
    all_reviews_res = supabase.table('post_reviews').select('post_id, category, selected_choices, rating').execute()

    all_users_df = pd.DataFrame(all_users_res.data).rename(columns={'id': 'user_id'})
    all_posts_df = pd.DataFrame(all_posts_res.data).rename(columns={'id': 'post_id'})

    # --- Xử lý Tương tác với logic Tích cực/Tiêu cực ---
    interaction_dfs = []

    # 1. Xử lý Lượt xem (Time-aware) - Luôn là tín hiệu dương
    views_df = pd.DataFrame(views_res.data)
    if not views_df.empty:
        def calculate_view_weight(duration_ms):
            if pd.isna(duration_ms) or duration_ms <= 0: return 1.0
            return 1.5 + math.log((duration_ms / 1000) + 1)
        views_df['weight'] = views_df['view_duration_ms'].apply(calculate_view_weight)
        interaction_dfs.append(views_df[['user_id', 'post_id', 'weight']])

    # 2. Xử lý Likes và Saves - Luôn là tín hiệu dương
    likes_df = pd.DataFrame(likes_res.data); likes_df['weight'] = 4.0
    saves_df = pd.DataFrame(saves_res.data); saves_df['weight'] = 5.0
    if not likes_df.empty: interaction_dfs.append(likes_df)
    if not saves_df.empty: interaction_dfs.append(saves_df)
    
    # 3. Xử lý Reviews (Cả Tích cực và Tiêu cực)
    reviews_df = pd.DataFrame(all_reviews_res.data)
    if not reviews_df.empty:
        # Cần author_id để biết ai là người review
        posts_authors_df = all_posts_df[['post_id', 'author_id']].rename(columns={'author_id': 'user_id'})
        reviews_with_user_df = pd.merge(reviews_df, posts_authors_df, on='post_id')

        def calculate_review_weight(rating):
            if rating >= 4: return 7.0   # Rất tích cực
            if rating <= 2: return -5.0  # Rất tiêu cực
            return 0.0 # Rating 3 sao, trung tính, không đưa vào tương tác có trọng số

        reviews_with_user_df['weight'] = reviews_with_user_df['rating'].apply(calculate_review_weight)
        # Chỉ lấy các review có trọng số khác 0
        weighted_reviews = reviews_with_user_df[reviews_with_user_df['weight'] != 0.0]
        if not weighted_reviews.empty:
            interaction_dfs.append(weighted_reviews[['user_id', 'post_id', 'weight']])
    
    # Gộp tất cả các tương tác và xử lý trường hợp có cả tương tác dương và âm
    if not interaction_dfs:
        interactions_df = pd.DataFrame(columns=['user_id', 'post_id', 'weight'])
    else:
        # Thay vì lấy max, chúng ta sẽ SUM các trọng số.
        # Điều này cho phép tín hiệu tiêu cực "triệt tiêu" bớt tín hiệu tích cực.
        full_interactions_df = pd.concat(interaction_dfs)
        interactions_df = full_interactions_df.groupby(['user_id', 'post_id'])['weight'].sum().reset_index()


    # --- Xử lý Đặc trưng Món ăn (Item Features) - Giữ lại tất cả ---
    print("Processing item features...")
    item_features_list = []
    
    # 1. Thêm food_category
    if not all_posts_df.empty:
        for _, row in all_posts_df.dropna(subset=['food_category']).iterrows():
            item_features_list.append({'post_id': row['post_id'], 'feature': f"category:{row['food_category']}"})
    
    # 2. Thêm TẤT CẢ review_choices làm feature
    if not reviews_df.empty:
        for _, row in reviews_df.iterrows():
            review_category = row['category']
            selected_choices = row.get('selected_choices')
            if selected_choices and isinstance(selected_choices, list):
                for choice in selected_choices:
                    feature_string = f"{review_category}:{choice}"
                    item_features_list.append({'post_id': row['post_id'], 'feature': feature_string})
    
    item_features_df = pd.DataFrame(item_features_list)
    
    print(f"Fetched {len(interactions_df)} interactions, {len(item_features_df)} item features for {len(all_users_df)} users.")
    return interactions_df, item_features_df, all_users_df, all_posts_df

# --- PHẦN 3: HUẤN LUYỆN MODEL ---
def train_model(interactions_df, item_features_df, all_users_df, all_posts_df):
    print("Preparing data and training model...")
    dataset = Dataset()
    
    all_possible_features = []
    if not item_features_df.empty:
        all_possible_features = item_features_df['feature'].unique()
        
    dataset.fit(
        users=all_users_df['user_id'].unique(),
        items=all_posts_df['post_id'].unique(),
        item_features=all_possible_features
    )

    (interactions_matrix, weights_matrix) = dataset.build_interactions(
        (row['user_id'], row['post_id'], row['weight']) for _, row in interactions_df.iterrows()
    )
    
    item_features_matrix = None
    if not item_features_df.empty:
        features_grouped = item_features_df.groupby('post_id')['feature'].apply(list)
        item_features_iterable = ((post_id, features) for post_id, features in features_grouped.items())
        item_features_matrix = dataset.build_item_features(item_features_iterable, normalize=True)

    # NÂNG CẤP: Sử dụng loss 'logistic' phù hợp hơn cho cả trọng số âm và dương
    model = LightFM(loss='logistic', no_components=30, learning_rate=0.05, random_state=42)
    model.fit(weights_matrix, item_features=item_features_matrix, epochs=20, verbose=True)
    print("DEBUG: Model training has finished successfully.")
    
    return model, dataset, item_features_matrix, interactions_matrix

# --- PHẦN 4: TẠO GỢI Ý ---
def generate_and_load(supabase: Client, model: LightFM, dataset: Dataset, item_features_matrix, interactions_matrix, all_users_df):
    print("Generating and loading recommendations...")
    user_id_map, _, item_id_map, _ = dataset.mapping()
    all_item_indices = np.arange(len(item_id_map))
    recommendations_to_upsert = []

    REORDER_PENALTY = -999.0

    for user_id in all_users_df['user_id']:
        user_index = user_id_map.get(user_id)
        if user_index is None: continue

        scores = model.predict(user_index, all_item_indices, item_features=item_features_matrix)
        
        if user_index < interactions_matrix.shape[0]:
            # Chỉ phạt những món có tương tác TÍCH CỰC
            user_interactions = interactions_matrix.tocsr()[user_index]
            positive_indices = user_interactions.indices[user_interactions.data > 0]
            scores[positive_indices] = REORDER_PENALTY

        top_indices = np.argsort(-scores)[:100]
        
        for item_index in top_indices:
            post_id = list(item_id_map.keys())[item_index]
            recommendations_to_upsert.append({
                'user_id': user_id, 
                'post_id': post_id, 
                'score': float(scores[item_index]), 
                'model_version': 'lightfm_v6.0_sentiment_aware'
            })

    if recommendations_to_upsert:
        print(f"Upserting {len(recommendations_to_upsert)} recommendations...")
        supabase.table('user_post_recommendations').delete().eq('model_version', 'lightfm_v6.0_sentiment_aware').execute()
        supabase.table('user_post_recommendations').upsert(recommendations_to_upsert).execute()
        print("Upsert completed.")

# --- PHẦN 5: HÀM MAIN ĐỂ CHẠY ---
if __name__ == '__main__':
    supabase_client = get_supabase_client()
    interactions, item_features, all_users, all_posts = fetch_data(supabase_client)
    
    # Chỉ chạy nếu có dữ liệu user và post
    if not all_users.empty and not all_posts.empty:
        trained_model, data_dataset, features_mat, interactions_mat = train_model(interactions, item_features, all_users, all_posts)
        generate_and_load(supabase_client, trained_model, data_dataset, features_mat, interactions_mat, all_users)
    else:
        print("No users or posts found. Skipping training.")