create extension if not exists pgcrypto;

create table if not exists public.app_users (
    id uuid primary key default gen_random_uuid(),
    external_user_id text unique,
    auth_user_id uuid unique,
    display_name text,
    email text unique,
    avatar_url text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.movies (
    watchmode_id bigint primary key,
    title text not null,
    movie_type text,
    release_year integer,
    release_date date,
    description text,
    poster_url text,
    backdrop_url text,
    imdb_id text,
    tmdb_id text,
    user_rating numeric,
    critic_score integer,
    runtime_minutes integer,
    genres jsonb not null default '[]'::jsonb,
    api_payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.reviews (
    id bigint generated always as identity primary key,
    watchmode_id bigint not null references public.movies(watchmode_id) on delete cascade,
    user_id uuid references public.app_users(id) on delete set null,
    review_text text not null,
    sentiment smallint not null check (sentiment in (0, 1)),
    sentiment_label text generated always as (
        case when sentiment = 1 then 'positive' else 'negative' end
    ) stored,
    rating smallint check (rating between 1 and 5),
    is_anonymous boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_app_users_external_user_id on public.app_users(external_user_id);
create index if not exists idx_app_users_auth_user_id on public.app_users(auth_user_id);
create index if not exists idx_reviews_watchmode_id on public.reviews(watchmode_id);
create index if not exists idx_reviews_user_id on public.reviews(user_id);
create index if not exists idx_reviews_created_at on public.reviews(created_at desc);
create index if not exists idx_movies_release_year on public.movies(release_year desc);
create index if not exists idx_movies_updated_at on public.movies(updated_at desc);
create index if not exists idx_movies_genres on public.movies using gin(genres);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_app_users_updated_at on public.app_users;
create trigger trg_app_users_updated_at
before update on public.app_users
for each row
execute function public.set_updated_at();

drop trigger if exists trg_movies_updated_at on public.movies;
create trigger trg_movies_updated_at
before update on public.movies
for each row
execute function public.set_updated_at();

drop trigger if exists trg_reviews_updated_at on public.reviews;
create trigger trg_reviews_updated_at
before update on public.reviews
for each row
execute function public.set_updated_at();

create or replace function public.handle_new_auth_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.app_users (auth_user_id, email, display_name, avatar_url)
    values (
        new.id,
        new.email,
        coalesce(new.raw_user_meta_data ->> 'display_name', new.raw_user_meta_data ->> 'name'),
        new.raw_user_meta_data ->> 'avatar_url'
    )
    on conflict (auth_user_id) do update
    set
        email = excluded.email,
        display_name = coalesce(excluded.display_name, public.app_users.display_name),
        avatar_url = coalesce(excluded.avatar_url, public.app_users.avatar_url),
        updated_at = now();

    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row
execute function public.handle_new_auth_user();

create or replace view public.movie_review_stats as
select
    m.watchmode_id,
    count(r.id)::int as review_count,
    avg(r.sentiment)::float as average_sentiment,
    avg(r.rating)::float as average_rating,
    coalesce(sum(case when r.sentiment = 1 then 1 else 0 end), 0)::int as positive_reviews,
    coalesce(sum(case when r.sentiment = 0 then 1 else 0 end), 0)::int as negative_reviews
from public.movies m
left join public.reviews r on r.watchmode_id = m.watchmode_id
group by m.watchmode_id;

create or replace view public.global_review_stats as
select
    count(*)::int as total_reviews,
    count(distinct watchmode_id)::int as reviewed_movies,
    avg(sentiment)::float as overall_sentiment,
    avg(rating)::float as overall_rating
from public.reviews;

alter table public.app_users enable row level security;
alter table public.movies enable row level security;
alter table public.reviews enable row level security;

drop policy if exists "movies_are_public_read" on public.movies;
create policy "movies_are_public_read"
on public.movies
for select
to anon, authenticated
using (true);

drop policy if exists "reviews_are_public_read" on public.reviews;
create policy "reviews_are_public_read"
on public.reviews
for select
to anon, authenticated
using (true);

drop policy if exists "users_can_read_own_profile" on public.app_users;
create policy "users_can_read_own_profile"
on public.app_users
for select
to authenticated
using (auth.uid() = auth_user_id);

drop policy if exists "users_can_update_own_profile" on public.app_users;
create policy "users_can_update_own_profile"
on public.app_users
for update
to authenticated
using (auth.uid() = auth_user_id)
with check (auth.uid() = auth_user_id);

drop policy if exists "authenticated_users_can_insert_reviews" on public.reviews;
create policy "authenticated_users_can_insert_reviews"
on public.reviews
for insert
to authenticated
with check (
    user_id in (
        select id
        from public.app_users
        where auth_user_id = auth.uid()
    )
    and is_anonymous = false
);

drop policy if exists "anonymous_reviews_via_service_role_only" on public.reviews;
create policy "anonymous_reviews_via_service_role_only"
on public.reviews
for insert
to anon
with check (false);

drop policy if exists "service_role_manage_movies" on public.movies;
create policy "service_role_manage_movies"
on public.movies
for all
to service_role
using (true)
with check (true);

drop policy if exists "service_role_manage_reviews" on public.reviews;
create policy "service_role_manage_reviews"
on public.reviews
for all
to service_role
using (true)
with check (true);

drop policy if exists "service_role_manage_app_users" on public.app_users;
create policy "service_role_manage_app_users"
on public.app_users
for all
to service_role
using (true)
with check (true);
