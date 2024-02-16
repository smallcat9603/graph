CREATE CONSTRAINT Ingredient_name IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE;
CREATE CONSTRAINT Dish_id IF NOT EXISTS FOR (d:Dish) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT DishType_name IF NOT EXISTS FOR (d:DishType) REQUIRE d.name IS UNIQUE;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/smallcat9603/graph/main/data/newfood.csv" AS row 
CREATE (d:Dish {id:row.id}) 
SET d += apoc.map.clean(row, ['id','dishTypes','ingredients'],[]) 
FOREACH (i in split(row.ingredients,',') | MERGE (in:Ingredient {name:toLower(replace(i,'-',' '))}) MERGE (in)<-[:HAS_INGREDIENT]-(d)) 
FOREACH (dt in split(row.dishTypes,',')  | MERGE (dts:DishType {name:dt}) MERGE (dts)<-[:DISH_TYPE]-(d));
