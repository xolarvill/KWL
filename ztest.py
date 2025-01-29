from geopy.geocoders import Nominatim
from geopy.distance import geodesic


geolocator = Nominatim(user_agent="geopyroutine")
location1 = geolocator.geocode("北京市")
print((location1.latitude, location1.longitude))
location2 = geolocator.geocode("杭州市")
print((location2.latitude, location2.longitude))
distance = geodesic(
    (location1.latitude, location1.longitude), 
    (location2.latitude, location2.longitude)).km
print(f"北京到上海的直线距离：{distance:.2f}公里")  # 输出：约1068公里