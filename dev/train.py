import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from lightfm import LightFM
from lightfm.data import Dataset

# =================================================================================
# SCRIPT HUẤN LUYỆN MODEL GỢI Ý - PHIÊN BẢN HOÀN THIỆN (V4.0 - FULL METADATA)
# - Tận dụng TOÀN BỘ metadata: FoodCategory và ReviewChoice.
# =================================================================================


# --- PHẦN 1: KẾT NỐI VÀ TRÍCH XUẤT DỮ LIỆU (EXTRACT) ---
def get_supabase_client() -> Client:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in .env file")
    return create_client(url, key)

def fetch_data(supabase: Client):
    print("Fetching all data from Supabase...")
    
    # Lấy dữ liệu chính
    all_users_res = supabase.table('profiles').select('id').execute()
    all_posts_res = supabase.table('posts').select('id, food_category').execute()
    
    # Lấy các loại tương tác
    views_res = supabase.table('post_views').select('user_id, post_id').execute()
    likes_res = supabase.table('post_likes').select('user_id, post_id').execute()
    saves_res = supabase.table('post_saves').select('user_id, post_id').execute()
    explicit_reviews_res = supabase.table('post_reviews').select('post_id, rating').filter('rating', 'gte', 4).execute()
    
    # <<<--- NÂNG CẤP: Lấy dữ liệu đặc trưng từ review_choices ---
    review_choices_res = supabase.table('post_reviews').select('post_id, category, selected_choices').execute()
    # ---------------------------------------------------------->

    # Xử lý Tương tác
    all_users_df = pd.DataFrame(all_users_res.data).rename(columns={'id': 'user_id'})
    all_posts_df = pd.DataFrame(all_posts_res.data).rename(columns={'id': 'post_id'})

    views_df = pd.DataFrame(views_res.data); views_df['weight'] = 1.0
    likes_df = pd.DataFrame(likes_res.data); likes_df['weight'] = 2.0
    saves_df = pd.DataFrame(saves_res.data); saves_df['weight'] = 3.0
    
    reviews_df = pd.DataFrame(explicit_reviews_res.data)
    posts_authors_df = pd.DataFrame(supabase.table('posts').select('id, author_id').execute().data).rename(columns={'id': 'post_id', 'author_id': 'user_id'})
    if not reviews_df.empty and not posts_authors_df.empty:
        reviews_df = pd.merge(reviews_df, posts_authors_df, on='post_id')
        reviews_df['weight'] = 5.0
        reviews_df = reviews_df[['user_id', 'post_id', 'weight']]
    else:
        reviews_df = pd.DataFrame(columns=['user_id', 'post_id', 'weight'])

    interactions_df = pd.concat([views_df, likes_df, saves_df, reviews_df])
    if not interactions_df.empty:
        interactions_df = interactions_df.sort_values('weight', ascending=False).drop_duplicates(['user_id', 'post_id'])

    # --- Xử lý Đặc trưng Món ăn (Item Features) ---
    print("Processing item features...")
    item_features_list = []
    
    # 1. Thêm food_category làm feature
    if not all_posts_df.empty:
        for _, row in all_posts_df.dropna(subset=['food_category']).iterrows():
            # Tạo feature có dạng 'category:vietnameseNoodles'
            item_features_list.append({'post_id': row['post_id'], 'feature': f"category:{row['food_category']}"})
    
    # <<<--- NÂNG CẤP: Thêm review_choices làm feature ---
    review_choices_df = pd.DataFrame(review_choices_res.data)
    if not review_choices_df.empty:
        for _, row in review_choices_df.iterrows():
            review_category = row['category'] # ví dụ: 'food', 'ambiance'
            selected_choices = row.get('selected_choices', []) # list các string, ví dụ: ['foodFlavorful', 'foodFreshIngredients']
            
            if selected_choices: # Chỉ xử lý nếu danh sách không rỗng
                for choice in selected_choices:
                    # Tạo feature có dạng 'food:foodFlavorful', 'ambiance:ambianceCozy'
                    feature_string = f"{review_category}:{choice}"
                    item_features_list.append({'post_id': row['post_id'], 'feature': feature_string})
    # ----------------------------------------------------->
    
    item_features_df = pd.DataFrame(item_features_list)
    
    print(f"Fetched {len(interactions_df)} interactions, {len(item_features_df)} item features for {len(all_users_df)} users.")
    return interactions_df, item_features_df, all_users_df, all_posts_df


# --- CÁC HÀM train_model và generate_and_load KHÔNG CẦN THAY ĐỔI ---
# Chúng đã được thiết kế để nhận vào một danh sách đặc trưng chung
# Bạn chỉ cần đảm bảo hàm fetch_data chuẩn bị đúng item_features_df

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
        (row['user_id'], row['post_id'], row['weight']) for index, row in interactions_df.iterrows()
    )
    
    item_features_matrix = None
    if not item_features_df.empty:
        features_grouped = item_features_df.groupby('post_id')['feature'].apply(list)
        item_features_iterable = ((post_id, features) for post_id, features in features_grouped.items())
        item_features_matrix = dataset.build_item_features(item_features_iterable, normalize=True)

    model = LightFM(loss='warp', no_components=30, learning_rate=0.05, random_state=42)
    model.fit(weights_matrix, item_features=item_features_matrix, epochs=20, verbose=True)
    print("DEBUG: Model training has finished successfully.")
    
    return model, dataset, item_features_matrix, interactions_matrix


def generate_and_load(supabase: Client, model: LightFM, dataset: Dataset, item_features_matrix, interactions_matrix, all_users_df):
    print("Generating and loading recommendations...")
    user_id_map, _, item_id_map, _ = dataset.mapping()
    all_item_indices = np.arange(len(item_id_map))
    recommendations_to_upsert = []

    for user_id in all_users_df['user_id']:
        user_index = user_id_map.get(user_id)
        if user_index is None:
            continue

        scores = model.predict(user_index, all_item_indices, item_features=item_features_matrix)
        
        if user_index < interactions_matrix.shape[0]:
            known_positives_indices = interactions_matrix.tocsr()[user_index].indices
            scores[known_positives_indices] = -np.inf

        top_indices = np.argsort(-scores)[:100]
        
        for item_index in top_indices:
            if scores[item_index] > -np.inf:
                post_id = list(item_id_map.keys())[item_index]
                recommendations_to_upsert.append({
                    'user_id': user_id, 
                    'post_id': post_id, 
                    'score': float(scores[item_index]), 
                    'model_version': 'lightfm_v4.0_full_meta' # Cập nhật phiên bản
                })

    if recommendations_to_upsert:
        print(f"Upserting {len(recommendations_to_upsert)} recommendations...")
        supabase.table('user_post_recommendations').delete().eq('model_version', 'lightfm_v4.0_full_meta').execute()
        supabase.table('user_post_recommendations').upsert(recommendations_to_upsert).execute()
        print("Upsert completed.")


if __name__ == '__main__':
    supabase_client = get_supabase_client()
    interactions, item_features, all_users, all_posts = fetch_data(supabase_client)
    
    if not interactions.empty:
        trained_model, data_dataset, features_mat, interactions_mat = train_model(interactions, item_features, all_users, all_posts)
        generate_and_load(supabase_client, trained_model, data_dataset, features_mat, interactions_mat, all_users)
    else:
        print("No interaction data found. Skipping training.")